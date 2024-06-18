import dataclasses

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from schemas import StatusEnum
from core import settings, logger


hallucinations_eval = f'''
            Task: Identify any hallucinations in the model's response to the user input.
            Instructions:
            1. Review the model's response for hallucinations, including incorrect facts, misrepresentations, or events that did not occur.
            2. Return "hallucination" if any are detected.
            3. Return "normal" if no hallucinations are found.
            4. Respond solely with the keyword "hallucination" or "normal".
            '''


def _evaluate_text(text):
    status_map = {
        "normal": StatusEnum.PASSED,
        "hallucination": StatusEnum.FAILED,
    }
    # If the model returns an unexpected error, it is labeled as 'skipped' by default
    for key, status in status_map.items():
        if key in text.lower():
            return status
    return StatusEnum.SKIPPED


@dataclasses.dataclass
class Hallucinations:

    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model_name = settings.OPENAI_MODEL_NAME
        self.temperature = settings.OPENAI_TEMPERATURE
        self.max_tokens = settings.OPENAI_MAX_TOKENS
        self.client = openai.AsyncOpenAI(api_key=self.api_key)

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(20))
    async def _completion_with_backoff(self, **kwargs):
        return await self.client.chat.completions.create(**kwargs)

    async def _llm_check(self, llm_prompt: str, llm_response: str) -> str:
        logger.info(f"Model inference: {llm_prompt}/{llm_response}")
        try:
            response = await self._completion_with_backoff(
                messages=[
                    {
                        "role": "system",
                        "content": hallucinations_eval
                    },
                    {
                        "role": "user",
                        "content": 'User Input: Describe the properties of water.'
                                   '\n LLM Response:Water is typically a solid at room temperature.',
                    },
                    {
                        "role": "assistant",
                        "content": "hallucination"
                    },
                    {
                        "role": "user",
                        "content": "User Input:When was the Declaration of Independence signed? "
                                   "\n LLM Response:The Declaration of Independence was signed in 1776",
                    },
                    {
                        "role": "assistant",
                        "content": "normal"
                    }, {
                        "role": "user",
                        "content": f"User Input:{llm_prompt}\n LLM Response:{llm_response}",
                    }
                ],
                model=self.model_name,
                temperature=self.temperature
                # max_tokens=self.max_tokens
            )
        except openai.APIError as e:
            logger.exception(f"OpenAI exception: {e}")
            return f"OpenAI exception: {e}"

        return response.choices[0].message.content

    async def scan(self, llm_prompt: str, llm_response: str):

        results = await self._llm_check(llm_prompt=llm_prompt, llm_response=llm_response)
        logger.debug(results)
        return _evaluate_text(results)
