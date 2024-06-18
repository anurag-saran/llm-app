from core import logger  # type: ignore
from .base_client import BaseLlmClient
from .huggingface_client import HuggingFaceClient
from .openai_client import OpenAiClient

LLM_TYPES = {
    "openai": OpenAiClient,
    "huggingface": HuggingFaceClient,
}


def llm_factory(llm_type: str, model_name: str):
    llm_class = LLM_TYPES.get(llm_type)

    if llm_class:
        logger.info(f"Running {llm_type}/{model_name}")
        return llm_class(model_name)

    raise NotImplementedError(f"LLM {llm_type}/{model_name} is not supported")
