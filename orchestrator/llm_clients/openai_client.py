import openai

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from .base_client import BaseLlmClient
from core import logger, settings  # type: ignore


class OpenAiClient(BaseLlmClient):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)

        self.api_key = settings.OPENAI_API_KEY
        self.temperature = kwargs.get("temperature", settings.OPENAI_TEMPERATURE)

        self.client = openai.AsyncOpenAI(api_key=self.api_key)

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(20))
    async def _completion_with_backoff(self, **kwargs):
        return await self.client.chat.completions.create(**kwargs)

    async def inference(self, prompt: str, max_tokens: int | None = settings.OPENAI_MAX_TOKENS):
        logger.debug(f"Model inference: {prompt}")

        if not max_tokens:
            max_tokens = settings.OPENAI_MAX_TOKENS

        try:
            response = await self._completion_with_backoff(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=max_tokens
            )
        except openai.APIError as e:
            logger.exception(f"OpenAI exception: {e}")
            return

        return response.choices[0].message.content
