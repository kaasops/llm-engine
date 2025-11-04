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
# Third-party imports
from starlette.responses import StreamingResponse

# Standard library imports
import json
from typing import Any, Dict, Union

# Try to import orjson for better performance, fallback to standard json
try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False


class JSONUtils:
    """Utility class for JSON operations with enhanced error handling and performance"""
    
    @staticmethod
    def serialize(data: Any, ensure_ascii: bool = False) -> str:
        """
        Serialize data to JSON string with error handling
        
        Args:
            data: Data to serialize
            ensure_ascii: Whether to escape non-ASCII characters
            
        Returns:
            JSON string
            
        Raises:
            TypeError: If data is not JSON serializable
        """
        try:
            # Use orjson for better performance if available
            if HAS_ORJSON:
                # orjson returns bytes, so we need to decode
                return orjson.dumps(data).decode('utf-8')
            else:
                return json.dumps(data, ensure_ascii=ensure_ascii)
        except (TypeError, ValueError) as e:
            logger.error(f"JSON serialization failed: {str(e)}")
            raise TypeError(f"Failed to serialize data to JSON: {str(e)}")
    
    @staticmethod
    def deserialize(json_str: str) -> Union[Dict, list, str, int, float, bool, None]:
        """
        Deserialize JSON string with error handling
        
        Args:
            json_str: JSON string to deserialize
            
        Returns:
            Deserialized data
            
        Raises:
            json.JSONDecodeError: If JSON string is invalid
        """
        try:
            # Use orjson for better performance if available
            if HAS_ORJSON:
                return orjson.loads(json_str)
            else:
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"JSON deserialization failed: {str(e)}")
            # Convert orjson error to json.JSONDecodeError for consistency
            if HAS_ORJSON and isinstance(e, ValueError):
                raise json.JSONDecodeError(f"Failed to parse JSON: {str(e)}", json_str, 0)
            raise


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
    def get_models_from_env() -> list:
        """Get list of models from MODELS environment variable"""
        models_env = os.getenv("MODELS", "")
        if not models_env:
            raise ValueError("MODELS environment variable is not set")
        
        models = [model.strip() for model in models_env.split(",") if model.strip()]
        if not models:
            raise ValueError("No valid models found in MODELS environment variable")
        
        logger.info(f"Loaded models from environment: {models}")
        return models

class ModelManager:
    """Manages vLLM model lifecycle and operations"""
    def __init__(self):
        self.model_name = None
        self.model = None
        self.sampling_params = None
        self.s3_loader = None
        self.local_model_path = None
        
    @classmethod
    async def _initialize_model(cls, model_name: str):
        """Initialize the model with sleep mode enabled"""
        self = cls()
        self.model_name = model_name
        try:
            logger.info(f"Initializing model: {self.model_name}")
            
            # Check if model is from S3
            if self.model_name.startswith("s3://"):
                logger.info(f"Detected S3 model: {self.model_name}")
                self.s3_loader = S3ModelLoader()
                self.local_model_path = self.s3_loader.download_model_from_s3(self.model_name)
                # Use the local path for vLLM
                actual_model_path = self.local_model_path
            else:
                # Use HuggingFace model directly
                actual_model_path = self.model_name

            engine_args = AsyncEngineArgs(
                model=actual_model_path,
                enforce_eager=True,
                enable_sleep_mode=True,
            )
            self.model = AsyncLLM.from_engine_args(engine_args)
            self.sampling_params = SamplingParams(**ModelConfig.DEFAULT_SAMPLING_PARAMS)
            await self.model.reset_prefix_cache()
            await self.model.sleep(level=1)
            logger.info(f"Model {self.model_name} initialized successfully")
            if self.s3_loader:
                self.s3_loader.cleanup()
            return self
        except Exception as e:
            logger.error(f"Failed to initialize model {self.model_name}: {str(e)}")
            # Clean up S3 loader if it was created
            if self.s3_loader:
                self.s3_loader.cleanup()
            raise
    
    async def generate_chat_completion(self, messages: list[ChatMessage], custom_params: Optional[Dict[str, Any]] = None, request_id: str = "", stream: bool = False):
        """Generate chat completion with proper error handling and timing"""
        # Convert messages to prompt format
        prompt = self._format_messages_to_prompt(messages)
        
        # Wake up the model
        await self.model.wake_up()
        
        # Use custom sampling parameters if provided
        sampling_params = self.sampling_params
        if custom_params:
            sampling_params = SamplingParams(**custom_params)

        generated_text = ""
        async for output in self.model.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id):
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
                            "model": self.model_name,
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
                        yield f"data: {JSONUtils.serialize(chunk)}\n\n"
            # Check if generation is finished
            if output.finished:
                if stream:
                    # Send final chunk with finish_reason
                    final_chunk = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": self.model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }
                        ]
                    }
                    yield f"data: {JSONUtils.serialize(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                break
        
        if not stream:
            yield generated_text
            
    async def sleep_model_after_response(self):
        """Put model to sleep after response is sent"""
        try:
            await asyncio.sleep(0.1)  # Small delay to ensure response is sent
            await self.model.reset_prefix_cache()
            await self.model.sleep(level=1)
            logger.info(f"Model {self.model_name} put to sleep after response")
        except Exception as e:
            logger.error(f"Failed to put model {self.model_name} to sleep after response: {str(e)}")
    
    def _format_messages_to_prompt(self, messages: list[ChatMessage]) -> str:
        """Convert chat messages to a prompt string"""
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
    
    async def cleanup(self):
        """Clean up S3 model resources if applicable"""
        if self.s3_loader:
            self.s3_loader.cleanup()

# FastAPI application
api = FastAPI(
    title="vLLM API",
    description="Serving vLLM models through Ray Serve with OpenAPI docs. Includes OpenAI-compatible endpoints.",
    version="1.0.0"
)

@serve.deployment(ray_actor_options={"num_cpus": 1.0, "num_gpus": 1.0}, user_config={"model": "/home/ray/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/"})
@serve.ingress(api)
class LLMServingAPI:
    """Main API class for serving multiple LLM models"""
    
    def __init__(self):
        # Get models from environment variable
        self.models = ModelConfig.get_models_from_env()        
        self.model_managers = {}
    
    async def reconfigure(self, config: dict):
        # Create model managers for each model
        for model_name in self.models:
            try:
                self.model_managers[model_name] = await ModelManager._initialize_model(model_name)
                logger.info(f"Successfully initialized model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize model {model_name}: {str(e)}")
                raise

        logger.info(f"LLM Serving API initialized with models: {list(self.model_managers.keys())}")
    
    @api.get("/health")
    async def health_check(self) -> Dict[str, str]:
        """Health check endpoint"""
        return {"status": "healthy", "models": ",".join(self.models)}
    
    # OpenAI-compatible endpoints
    @api.post("/v1/chat/completions")
    async def create_chat_completion(self, request: ChatCompletionRequest, background_tasks: BackgroundTasks):
        """OpenAI-compatible chat completion endpoint"""
        try:
            # Check if requested model is available
            if request.model not in self.model_managers:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model {request.model} not found. Available models: {list(self.model_managers.keys())}"
                )
            
            # Prepare custom sampling parameters
            custom_params = {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens,
                "output_kind": RequestOutputKind.DELTA
            }

            request_id = random_uuid()

            if request.stream:
                # Create a generator that includes sleep after streaming is complete
                async def stream_with_sleep():
                    model_manager = self.model_managers[request.model]
                    try:
                        async for chunk in model_manager.generate_chat_completion(
                            request.messages, custom_params, request_id, stream=True
                        ):
                            yield chunk
                    finally:
                        # Put model to sleep after streaming is complete
                        await model_manager.sleep_model_after_response()
                
                # Return streaming response in OpenAI format
                return StreamingResponse(
                    stream_with_sleep(),
                    media_type="text/event-stream"
                )
            
            # Generate non-streaming chat completion
            generated_text = ""
            async for text_chunk in self.model_managers[request.model].generate_chat_completion(
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
            
            # Schedule model sleep after response is sent
            background_tasks.add_task(self.model_managers[request.model].sleep_model_after_response)
            
            return response
            
        except Exception as e:
            logger.error(f"Chat completion failed for model {request.model}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")
    
    @api.get("/v1/models")
    async def list_openai_models(self) -> Dict[str, Any]:
        """OpenAI-compatible models list endpoint"""
        models_list = []
        for model_name in self.models:
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


# Ray Serve deployment
app = LLMServingAPI.bind()