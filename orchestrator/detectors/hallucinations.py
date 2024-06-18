import httpx

from .base_detector import BaseDetector, DetectorMode
from core import logger, settings


class HallucinationsDetector(BaseDetector):
    detector_type = "hallucinations"

    def __init__(self):
        super().__init__()
        if not settings.DETECTOR_API_RELEVANCE_URL:
            # @TODO: add our own exceptions
            raise EnvironmentError("DETECTOR_API_RELEVANCE_URL is not set")

    async def evaluate(self, llm_prompt, llm_response, metadata=None, mode=DetectorMode.model_output):
        logger.debug(f"Evaluating hallucinations on LLM response: {llm_response}")

        headers = {
            settings.DETECTORS_API_KEY_NAME: settings.DETECTORS_API_KEY
        }
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                settings.DETECTOR_API_HALLUCINATIONS_URL,
                json={"llm_prompt": llm_prompt, "llm_response": llm_response},
                headers=headers
            )
            return response.json()

    @staticmethod
    async def health_check():
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                response = await client.get(f"{settings.DETECTOR_API_HALLUCINATIONS_URL}/health")
                if response.status_code == httpx.codes.OK:
                    return response.json()
                else:
                    return {"health": "error", "status_code": response.status_code}
            except httpx.HTTPError as e:
                return {"health": "error", "error": str(e)}
            except Exception as e:
                return {"health": "error", "error": str(e)}
