from ray import serve
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import logging
from typing import Optional, Dict, Any
import time
import asyncio
from s3_model_loader import S3ModelLoader
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.sampling_params import RequestOutputKind
from vllm.utils import random_uuid
from starlette.responses import StreamingResponse
from typing import TYPE_CHECKING, AsyncGenerator
import json
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# OpenAI-compatible schemas
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

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage

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
        
        # Extract model_id and model_source from each model configuration
        models = {}
        for model_config in models_config:
            if isinstance(model_config, dict) and "model_source" in model_config:
                model_id = model_config.get("model_id", model_config["model_source"].split("/")[-1])
                models[model_id] = model_config["model_source"]
            elif isinstance(model_config, str):
                # Handle case where model is just a string path
                model_id = model_config.split("/")[-1]
                models[model_id] = model_config
            else:
                logger.warning(f"Skipping invalid model configuration: {model_config}")
        
        if not models:
            raise ValueError("No valid models found in user_config")
        
        logger.info(f"Loaded model names and paths from user_config: {models}")
        return models
    
class ModelManager:
    """Manages vLLM model lifecycle and operations"""
    def __init__(self):
        self.model_source = None
        self.model_id = None
        self.engine = None
        self.sampling_params = None
        self.s3_loader = None
        self.local_model_path = None
        self.active_requests = 0
        self.sleep_lock = asyncio.Lock()
        self.request_lock = asyncio.Lock()
        self.tokenizer = None
        self.has_chat_template = False
        
    @classmethod
    async def start(cls, model_id: str, model_source: str):
        """Initialize the model with sleep mode enabled"""
        self = cls()
        self.model_source = model_source
        self.model_id = model_id
        try:
            logger.info(f"Initializing model: {self.model_source}")
            
            # Check if model is from S3
            if self.model_source.startswith("s3://"):
                logger.info(f"Detected S3 model: {self.model_source}")
                self.s3_loader = S3ModelLoader()
                self.local_model_path = self.s3_loader.download_model_from_s3(self.model_source)
                # Use the local path for vLLM
                actual_model_path = self.local_model_path
            else:
                # Use HuggingFace model directly
                actual_model_path = self.model_source

            engine_args = AsyncEngineArgs(
                model=actual_model_path,
                enforce_eager=True,
                enable_sleep_mode=True,
                # load_format="runai_streamer"
            )
            self.engine = AsyncLLM.from_engine_args(engine_args)
            self.sampling_params = SamplingParams(**ModelConfig.DEFAULT_SAMPLING_PARAMS)
            
            # Load tokenizer for chat template support
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(actual_model_path)
                self.has_chat_template = hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None
                if self.has_chat_template:
                    logger.info(f"Model {self.model_id} has chat template support")
                else:
                    logger.info(f"Model {self.model_id} does not have a chat template, will use fallback formatting")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer for chat template: {str(e)}. Will use fallback formatting.")
                self.has_chat_template = False
            
            await self.engine.reset_prefix_cache()
            await self.engine.sleep(level=1)
            logger.info(f"Model {self.model_source} initialized successfully")
            if self.s3_loader:
                self.s3_loader.cleanup()
            return self
        except Exception as e:
            logger.error(f"Failed to initialize model {self.model_source}: {str(e)}")
            # Clean up S3 loader if it was created
            if self.s3_loader:
                self.s3_loader.cleanup()
            raise
    
    async def sleep_model_after_response(self):
        """Put model to sleep after response is sent, but only if no active requests"""
        async with self.request_lock:
            self.active_requests -= 1
            logger.info(f"Request completed. Active requests for {self.model_id}: {self.active_requests}")
            
            # Only put model to sleep if there are no active requests
            if self.active_requests <= 0:
                try:
                    await self.engine.reset_prefix_cache()
                    await self.engine.sleep(level=1)
                    logger.info(f"Model {self.model_id} put to sleep after response (no active requests)")
                except Exception as e:
                    logger.error(f"Failed to put model {self.model_id} to sleep after response: {str(e)}")

    def _format_messages_to_prompt(self, messages: list[ChatMessage]) -> str:
        """Convert chat messages to a prompt string using model's chat template if available"""
        # Convert ChatMessage objects to dictionaries for the tokenizer
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        # Try to use the model's chat template if available
        if self.has_chat_template and self.tokenizer:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    message_dicts,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return prompt
            except Exception as e:
                logger.warning(f"Failed to apply chat template for {self.model_id}: {str(e)}. Using fallback formatting.")
        
        # Fallback to simple string formatting if chat template is not available or fails
        prompt = ""
        for message in messages:
            if message.role == "system":
                prompt += f"System: {message.content}\n\n"
            elif message.role == "user":
                prompt += f"User: {message.content}\n\n"
            elif message.role == "assistant":
                prompt += f"Assistant: {message.content}\n\n"
        
        # Add the final assistant prompt
        prompt += "Assistant:"
        return prompt

    async def chat(self, messages: list[ChatMessage], custom_params: Optional[Dict[str, Any]] = None, request_id: str = "", stream: bool = False) -> AsyncGenerator:
        # Increment active requests counter
        async with self.request_lock:
            self.active_requests += 1
            logger.info(f"New request started. Active requests for {self.model_id}: {self.active_requests}")
        
        try:
            # Check if model is in sleep mode before waking up
            if await self.engine.is_sleeping():
                await self.engine.wake_up()
                logger.info(f"Model {self.model_id} woke up from sleep")
            else:
                # Model is already awake, no need to wake up
                pass
        except:
            logger.error("Failed to wake up model")

        try:
            # Convert messages to prompt format
            prompt = self._format_messages_to_prompt(messages)
            
            # Use custom sampling parameters if provided
            sampling_params = self.sampling_params
            if custom_params:
                sampling_params = SamplingParams(**custom_params)

            generated_text = ""
            async for output in self.engine.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id):
                for completion in output.outputs:
                    # In DELTA mode, we get only new tokens generated since last iteration
                    new_text = completion.text
                    if new_text:
                        generated_text += new_text
                        if stream:
                            # Format as OpenAI streaming response
                            chunk = {
                                "id": f"chatcmpl-{request_id}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": self.model_id,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "content": new_text
                                        },
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                # Check if generation is finished
                if output.finished:
                    if stream:
                        # Send final chunk with finish_reason
                        final_chunk = {
                            "id": f"chatcmpl-{request_id}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": self.model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }
                            ]
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                    break
            
            if not stream:
                yield generated_text
        except Exception as e:
            # Ensure we decrement the counter even if there's an error during streaming
            async with self.request_lock:
                self.active_requests -= 1
                logger.error(f"Error during chat streaming for {self.model_id}: {str(e)}")
            raise

# FastAPI application
api = FastAPI(
    title="vLLM API",
    description="Serving vLLM models through Ray Serve with OpenAPI docs. Includes OpenAI-compatible endpoints.",
    version="1.0.0"
)
 
@serve.deployment(ray_actor_options={"num_cpus": 1.0, "num_gpus": 1.0},user_config={"models": [{"model_source": "openai/gpt-oss-20b"}]})
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
    
    async def decrement_consumer_count(self, model_id: str):
        """Decrement the consumer count for a specific model"""
        async with self.consumer_lock:
            if model_id in self.active_consumers:
                self.active_consumers[model_id] -= 1
                logger.info(f"Decremented consumer count for {model_id}: {self.active_consumers[model_id]}")
                
                # If no more active consumers, we could optionally clean up resources
                if self.active_consumers[model_id] <= 0:
                    self.active_consumers[model_id] = 0
                    logger.info(f"No more active consumers for {model_id}")
                    self.current_active_model = None
            else:
                logger.warning(f"Attempted to decrement consumer count for unknown model: {model_id}")
    
    async def reconfigure(self, user_config: dict[str, Any]):
        # Get models from user_config
        self.models = ModelConfig.get_models_from_user_config(user_config)
        
        # Create model managers for each model
        for model_id, model_source in self.models.items():
            try:
                # Check if ModelManager for this model is already initialized
                if model_id in self.model_managers:
                    logger.info(f"Model {model_id} already initialized, skipping initialization")
                    continue
                
                self.model_managers[model_id] = await ModelManager.start(model_id, model_source)
                logger.info(f"Successfully initialized model: {model_id} -> {model_source}")
            except Exception as e:
                logger.error(f"Failed to initialize model {model_id} ({model_source}): {str(e)}")
                raise

        logger.info(f"LLM Serving API initialized with models: {list(self.model_managers.keys())}")

    @api.get("/v1/models")
    async def list_openai_models(self) -> dict[str, Any]:
        """OpenAI-compatible models list endpoint"""
        models_list = []
        for model_id in self.models.keys():
            models_list.append({
                "id": model_id,
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
        try:
            custom_params = {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens,
                "output_kind": RequestOutputKind.DELTA
            }

            request_id = random_uuid()

            if request.stream:
                # Create a generator for streaming response
                async def stream_generator():
                    model_manager = self.model_managers[request.model]
                    async for chunk in model_manager.chat(
                        request.messages, custom_params, request_id, stream=True
                    ):
                        yield chunk
                
                # Return streaming response in OpenAI format
                return StreamingResponse(
                    stream_generator(),
                    media_type="text/event-stream"
                )
            
            # Generate non-streaming chat completion
            generated_text = ""
            async for text_chunk in self.model_managers[request.model].chat(
                request.messages, custom_params, request_id, stream=False
            ):
                generated_text += text_chunk
            
            tokens_generated = len(generated_text.split())  # Rough estimate
            
            # Create response in OpenAI format
            response_message = ChatMessage(role="assistant", content=generated_text)
            choice = ChatCompletionChoice(
                index=0,
                message=response_message,
                finish_reason="stop"
            )
            
            usage = ChatCompletionUsage(
                prompt_tokens=0,  # Note: vLLM doesn't provide prompt token count easily
                completion_tokens=tokens_generated,
                total_tokens=tokens_generated
            )
            
            response = ChatCompletionResponse(
                id=f"chatcmpl-{request_id}",
                created=int(time.time()),
                model=request.model,
                choices=[choice],
                usage=usage
            )

            return response
        except Exception as e:
            logger.error(f"Chat completion failed for model {request.model}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")
        finally:
            # Add background tasks only after streaming is complete
            background_tasks.add_task(model_manager.sleep_model_after_response)
            background_tasks.add_task(self.decrement_consumer_count, request.model)

# Ray Serve deployment
app = LLMServingAPI.bind()