from ray import serve
from fastapi import FastAPI, HTTPException, BackgroundTasks
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from starlette.responses import StreamingResponse
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


api = FastAPI(
    title="vLLM API",
    description="Serving vLLM model through Ray Serve with OpenAPI docs.",
    version="1.0.0"
)

@serve.deployment(ray_actor_options={"num_cpus": 4.0, "num_gpus": 1.0}, user_config={"model": "/home/ray/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/"})
@serve.ingress(api)
class LLMServingAPI:
    def __init__(self):
        self.prompt = "Hi Who are you?"
        self.sampling_params = SamplingParams(temperature=0.8, top_p=0.95, output_kind=RequestOutputKind.DELTA)
        self.engine = None

    async def reconfigure(self, config: dict):
        engine_args = AsyncEngineArgs(
            model="/home/ray/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/",
            enforce_eager=True,
            enable_sleep_mode=True,
        )
        self.model = config["model"]
        self.engine = AsyncLLM.from_engine_args(engine_args)
        await self.engine.sleep(level=1)

    async def sleep_model_after_response(self):
        """Put model to sleep after response is sent"""
        try:
            await asyncio.sleep(0.1)  # Small delay to ensure response is sent
            await self.engine.reset_prefix_cache()
            await self.engine.sleep(level=1)
            logger.info("Model put to sleep after response")
        except Exception as e:
            logger.error(f"Failed to put model to sleep after response: {e}")
            # Re-raise the exception to ensure the background task failure is visible
            raise

    async def stream_response(self):
        async for output in self.engine.generate(prompt=self.prompt, sampling_params=self.sampling_params, request_id="test"):
            for completion in output.outputs:
                # In DELTA mode, we get only new tokens generated since last iteration
                new_text = completion.text
                if new_text:
                    # Yield the text to StreamingResponse in proper format
                    yield f"{new_text}\n"
            # Check if generation is finished
            if output.finished:
                break
    
    @api.post("/qwen")
    async def handle_request(self, background_tasks: BackgroundTasks) -> StreamingResponse:
        # Ensure model is initialized and sleeping
        await self.engine.wake_up()
        background_tasks.add_task(self.sleep_model_after_response)
        return StreamingResponse(self.stream_response(), media_type="text/plain")

app = LLMServingAPI.bind()