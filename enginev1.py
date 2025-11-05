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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.engine = None
        self.sampling_params = None
        self.s3_loader = None
        self.local_model_path = None
        self.active_requests = 0
        self.sleep_lock = asyncio.Lock()
        
    @classmethod
    async def start(cls, model_path: str):
        """Initialize the model with sleep mode enabled"""
        self = cls()
        self.model_path = model_path
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

# FastAPI application
api = FastAPI(
    title="vLLM API",
    description="Serving vLLM models through Ray Serve with OpenAPI docs. Includes OpenAI-compatible endpoints.",
    version="1.0.0"
)

@serve.deployment(ray_actor_options={"num_cpus": 1.0, "num_gpus": 1.0}, user_config={
    "models": [
        {"model_name": "Qwen/Qwen2.5-0.5B-Instruct","model_path": "/home/ray/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/"},
        {"model_name": "Qwen/Qwen2.5-7B-Instruct","model_path": "/home/ray/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/"},
        ]
        })
@serve.ingress(api)
class LLMServingAPI:
    """Main API class for serving multiple LLM models"""
    
    def __init__(self):
        # Initialize empty models list - will be populated in reconfigure
        self.models = []
        self.model_managers = {}
    
    async def reconfigure(self, user_config: dict[str, Any]):
        # Get models from user_config
        self.models = ModelConfig.get_models_from_user_config(user_config)
        
        # Create model managers for each model
        for model_name, model_path in self.models.items():
            try:
                self.model_managers[model_name] = await ModelManager.start(model_path)
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

# Ray Serve deployment
app = LLMServingAPI.bind()