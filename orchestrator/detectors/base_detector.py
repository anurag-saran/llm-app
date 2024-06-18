from enum import Enum


class DetectorMode(str, Enum):
    model_input = "model_input"
    model_output = "model_output"
    combine = "combine"


class BaseDetector:
    detector_type = "base"

    def __init__(self):
        pass

    async def evaluate(self, llm_prompt, llm_response, metadata=None, mode=DetectorMode.model_output):
        raise NotImplementedError

    @staticmethod
    async def health_check():
        raise NotImplementedError
