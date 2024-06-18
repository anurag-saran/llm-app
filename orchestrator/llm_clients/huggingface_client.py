import aiohttp

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from huggingface_hub import AsyncInferenceClient, InferenceTimeoutError

from core import logger, settings  # type: ignore
from .base_client import BaseLlmClient


class HuggingFaceClient(BaseLlmClient):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)

        self.api_key = settings.HUGGINGFACE_API_KEY
        self.temperature = kwargs.get("temperature", settings.HUGGINGFACE_TEMPERATURE)

        try:
            model = settings.HUGGINGFACE_MODELS_MAPPING[model_name]
        except KeyError:
            logger.error(f"Huggingface model {model_name} don't supporting")
            raise NotImplementedError

        self.client = AsyncInferenceClient(
            model=model,
            token=settings.HUGGINGFACE_API_KEY,
            timeout=settings.HUGGINGFACE_TIMEOUT,
        )

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(20))
    async def _completion_with_backoff(self, **kwargs):
        return await self.client.text_generation(**kwargs)

    async def inference(self, prompt: str, max_tokens: int | None = settings.HUGGINGFACE_MAX_TOKENS):
        logger.debug(f"Model inference: {prompt}")

        if not max_tokens:
            max_tokens = settings.HUGGINGFACE_MAX_TOKENS

        try:
            response = await self._completion_with_backoff(
                prompt=prompt, temperature=self.temperature, max_new_tokens=max_tokens
            )
        except InferenceTimeoutError or aiohttp.ClientResponseError as e:
            logger.exception(f"Huggingface exception: {e}")
            return

        return response
