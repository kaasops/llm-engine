from ray import serve
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import logging
from typing import Optional, Dict, Any
import time
import os
import asyncio
from s3_model_loader import S3ModelLoader
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.sampling_params import RequestOutputKind
from vllm.utils import random_uuid
from starlette.responses import StreamingResponse
from typing import TYPE_CHECKING, AsyncGenerator
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

class ModelConfig:
    """Configuration class for model settings"""
    DEFAULT_SAMPLING_PARAMS = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 512,
        "output_kind": RequestOutputKind.DELTA
    }
    
    @staticmethod
    def get_models_from_user_config(user_config: dict[str, Any]) -> dict[str, str]:
        """Get dictionary of model names and paths from user_config dictionary"""
        if not user_config or "models" not in user_config:
            raise ValueError("user_config does not contain 'models' key")
        
        models_config = user_config["models"]
        if not isinstance(models_config, list):
            raise ValueError("'models' in user_config must be a list")
        
        # Extract model_name and model_path from each model configuration
        models = {}
        for model_config in models_config:
            if isinstance(model_config, dict) and "model_path" in model_config:
                model_name = model_config.get("model_name", model_config["model_path"].split("/")[-1])
                models[model_name] = model_config["model_path"]
            elif isinstance(model_config, str):
                # Handle case where model is just a string path
                model_name = model_config.split("/")[-1]
                models[model_name] = model_config
            else:
                logger.warning(f"Skipping invalid model configuration: {model_config}")
        
        if not models:
            raise ValueError("No valid models found in user_config")
        
        logger.info(f"Loaded model names and paths from user_config: {models}")
        return models
    
class ModelManager:
    """Manages vLLM model lifecycle and operations"""
    def __init__(self):
        self.model_path = None
        self.model_name = None
        self.engine = None
        self.sampling_params = None
        self.s3_loader = None
        self.local_model_path = None
        self.active_requests = 0
        self.sleep_lock = asyncio.Lock()
        self.request_lock = asyncio.Lock()
        
    @classmethod
    async def start(cls, model_name: str, model_path: str):
        """Initialize the model with sleep mode enabled"""
        self = cls()
        self.model_path = model_path
        self.model_name = model_name
        try:
            logger.info(f"Initializing model: {self.model_path}")
            
            # Check if model is from S3
            if self.model_path.startswith("s3://"):
                logger.info(f"Detected S3 model: {self.model_path}")
                self.s3_loader = S3ModelLoader()
                self.local_model_path = self.s3_loader.download_model_from_s3(self.model_path)
                # Use the local path for vLLM
                actual_model_path = self.local_model_path
            else:
                # Use HuggingFace model directly
                actual_model_path = self.model_path

            engine_args = AsyncEngineArgs(
                model=actual_model_path,
                enforce_eager=True,
                enable_sleep_mode=True,
            )
            self.engine = AsyncLLM.from_engine_args(engine_args)
            self.sampling_params = SamplingParams(**ModelConfig.DEFAULT_SAMPLING_PARAMS)
            await self.engine.reset_prefix_cache()
            await self.engine.sleep(level=1)
            logger.info(f"Model {self.model_path} initialized successfully")
            if self.s3_loader:
                self.s3_loader.cleanup()
            return self
        except Exception as e:
            logger.error(f"Failed to initialize model {self.model_path}: {str(e)}")
            # Clean up S3 loader if it was created
            if self.s3_loader:
                self.s3_loader.cleanup()
            raise
    
    async def sleep_model_after_response(self):
        """Put model to sleep after response is sent, but only if no active requests"""
        async with self.request_lock:
            self.active_requests -= 1
            logger.info(f"Request completed. Active requests for {self.model_name}: {self.active_requests}")
            
            # Only put model to sleep if there are no active requests
            if self.active_requests <= 0:
                try:
                    await self.engine.reset_prefix_cache()
                    await self.engine.sleep(level=1)
                    logger.info(f"Model {self.model_name} put to sleep after response (no active requests)")
                except Exception as e:
                    logger.error(f"Failed to put model {self.model_name} to sleep after response: {str(e)}")

    async def chat(self, request) -> AsyncGenerator:
        # Increment active requests counter
        async with self.request_lock:
            self.active_requests += 1
            logger.info(f"New request started. Active requests for {self.model_name}: {self.active_requests}")
        
        try:
            # Check if model is in sleep mode before waking up
            if await self.engine.is_sleeping():
                await self.engine.wake_up()
                logger.info(f"Model {self.model_name} woke up from sleep")
            else:
                # Model is already awake, no need to wake up
                pass
        except:
            logger.error("Failed to wake up model")
        
        text_resp = "cjslfk jcdlsk jcld  jfvkjfhd kfjhvdkfjvhdf klvjhdfklvjh vdfkjvhdfkjvhdf kjvhdfk jvhd bghjn bfghghhgh fvdlk dsdcsd bhhg bgh bvgffhbhgf vff"
        tokens = text_resp.split(" ")

        try:
            for i, token in enumerate(tokens):
                chunk = {
                    "id": i,
                    "object": "chat.completion.chunk",
                    "created": time.time(),
                    "model": "blah",
                    "choices": [{"delta": {"content": token + " "}}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(1)
            yield "data: [DONE]\n\n"
        except Exception as e:
            # Ensure we decrement the counter even if there's an error during streaming
            async with self.request_lock:
                self.active_requests -= 1
                logger.error(f"Error during chat streaming for {self.model_name}: {str(e)}")
            raise

# FastAPI application
api = FastAPI(
    title="vLLM API",
    description="Serving vLLM models through Ray Serve with OpenAPI docs. Includes OpenAI-compatible endpoints.",
    version="1.0.0"
)
 
@serve.deployment(ray_actor_options={"num_cpus": 1.0, "num_gpus": 1.0},user_config={
    "models": [
        {"model_name": "Qwen/Qwen2.5-0.5B-Instruct","model_path": "/home/ray/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/"},
        {"model_name": "Qwen/Qwen2.5-7B-Instruct","model_path": "/home/ray/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/"}]})
@serve.ingress(api)
class LLMServingAPI:
    """Main API class for serving multiple LLM models"""
    
    def __init__(self):
        # Initialize empty models list - will be populated in reconfigure
        self.models = []
        self.model_managers = {}
        self.current_active_model = None  # Track the currently active model
        self.active_consumers = {}  # Track active consumers for each model
        self.consumer_lock = asyncio.Lock()  # Lock for thread-safe consumer count updates
    
    async def wait_for_available_model(self, requested_model: str, max_wait_time: int = 60, wait_interval: float = 0.5) -> bool:
        # If no model is currently active or the requested model is the same as the current active model, no need to wait
        if self.current_active_model is None or self.current_active_model == requested_model:
            return True
            
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            if self.current_active_model is None:
                 return True
            logger.info(f"Waiting for model switch from {self.current_active_model} to {requested_model}...")
            # Wait for the specified interval
            await asyncio.sleep(wait_interval)
            elapsed_time += wait_interval
        
        # Timeout reached
        logger.error(f"Timeout waiting for model switch from {self.current_active_model} to {requested_model} after {max_wait_time} seconds")
        return False
    
    async def decrement_consumer_count(self, model_name: str):
        """Decrement the consumer count for a specific model"""
        async with self.consumer_lock:
            if model_name in self.active_consumers:
                self.active_consumers[model_name] -= 1
                logger.info(f"Decremented consumer count for {model_name}: {self.active_consumers[model_name]}")
                
                # If no more active consumers, we could optionally clean up resources
                if self.active_consumers[model_name] <= 0:
                    self.active_consumers[model_name] = 0
                    logger.info(f"No more active consumers for {model_name}")
                    self.current_active_model = None
            else:
                logger.warning(f"Attempted to decrement consumer count for unknown model: {model_name}")
    
    async def reconfigure(self, user_config: dict[str, Any]):
        # Get models from user_config
        self.models = ModelConfig.get_models_from_user_config(user_config)
        
        # Create model managers for each model
        for model_name, model_path in self.models.items():
            try:
                self.model_managers[model_name] = await ModelManager.start(model_name, model_path)
                logger.info(f"Successfully initialized model: {model_name} -> {model_path}")
            except Exception as e:
                logger.error(f"Failed to initialize model {model_name} ({model_path}): {str(e)}")
                raise

        logger.info(f"LLM Serving API initialized with models: {list(self.model_managers.keys())}")

    @api.get("/v1/models")
    async def list_openai_models(self) -> dict[str, Any]:
        """OpenAI-compatible models list endpoint"""
        models_list = []
        for model_name in self.models.keys():
            models_list.append({
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "vllm"
            })
        
        return {
            "object": "list",
            "data": models_list
        }

    @api.get("/health")
    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        return {"status": "healthy", "models": list(self.models.keys())}

    @api.post("/v1/chat/completions")
    async def chat(self, request: ChatCompletionRequest, background_tasks: BackgroundTasks):
        # Get the requested model manager
        if request.model not in self.model_managers:
            raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found")
            
        # Wait only when model is changed
        if not await self.wait_for_available_model(request.model):
            raise HTTPException(status_code=503, detail="Service temporarily unavailable - model switch timeout")
            
        model_manager = self.model_managers[request.model]
        logger.info(f"Set active manager {request.model}")
        self.current_active_model = request.model  # Update the current active model
        
        # Increment consumer count for the model
        async with self.consumer_lock:
            if request.model not in self.active_consumers:
                self.active_consumers[request.model] = 0
            self.active_consumers[request.model] += 1
            logger.info(f"Active consumers for {request.model}: {self.active_consumers[request.model]}")
        
        # Create a wrapper generator that handles cleanup after streaming
        async def chat_with_cleanup():
            try:
                async for chunk in model_manager.chat(request):
                    yield chunk
            finally:
                # Add background tasks only after streaming is complete
                background_tasks.add_task(model_manager.sleep_model_after_response)
                background_tasks.add_task(self.decrement_consumer_count, request.model)
        
        return StreamingResponse(
            chat_with_cleanup(),
            media_type="text/event-stream"
        )

# Ray Serve deployment
app = LLMServingAPI.bind()