import httpx

from .base_detector import BaseDetector, DetectorMode
from core import logger, settings


class ToxicityDetector(BaseDetector):
    detector_type = "toxicity"

    def __init__(self):
        super().__init__()
        if not settings.DETECTOR_API_TOXICITY_URL:
            # @TODO: add our own exceptions
            raise EnvironmentError("DETECTOR_API_TOXICITY_URL is not set")

    async def evaluate(self, llm_prompt, llm_response, metadata=None, mode=DetectorMode.model_output):

        logger.debug(f"Evaluating toxicity on LLM response: {llm_response}")

        if mode == DetectorMode.combine:
            text = f"{llm_prompt} {llm_response}"
        elif mode == DetectorMode.model_input:
            text = llm_prompt
        else:
            text = llm_response

        headers = {
            settings.DETECTORS_API_KEY_NAME: settings.DETECTORS_API_KEY
        }

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                settings.DETECTOR_API_TOXICITY_URL,
                json={"text": text},
                headers=headers
            )
            return response.json()

    @staticmethod
    async def health_check():
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                response = await client.get(f"{settings.DETECTOR_API_TOXICITY_URL}/health")
                if response.status_code == httpx.codes.OK:
                    return response.json()
                else:
                    return {"health": "error", "status_code": response.status_code}
            except httpx.HTTPError as e:
                return {"health": "error", "error": str(e)}
            except Exception as e:
                return {"health": "error", "error": str(e)}
