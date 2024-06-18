import json

import dotenv
from pydantic_settings import BaseSettings

dotenv.load_dotenv()


class Settings(BaseSettings):

    # Detectors
    DETECTORS_API_KEY: str = "actual_api_key_123"
    DETECTORS_API_KEY_NAME: str = "access_token"
    DETECTOR_API_TOXICITY_URL: str = "http://localhost"
    DETECTOR_API_RELEVANCE_URL: str = "http://localhost"
    DETECTOR_API_HALLUCINATIONS_URL: str = "http://localhost"
    DETECTOR_API_PRIVACY_URL: str = "http://localhost"
    STEREOTYPE_FAIRNESS_MAX_TOKENS: int = 4
    HALLUCINATION_PRIVACY_MAX_TOKENS: int = 100

    # OpenAI settings
    OPENAI_API_KEY: str
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_MAX_TOKENS: int = 100
    OPENAI_RPM: int = 500

    # Huggingface settings
    HUGGINGFACE_API_KEY: str = ""
    HUGGINGFACE_MODELS_MAPPING: dict = {}
    HUGGINGFACE_TEMPERATURE: float = 0.7
    HUGGINGFACE_TIMEOUT: float = 1200
    HUGGINGFACE_MAX_TOKENS: int = 20

    # LLM concurrently requests
    LLM_CONCURRENTLY_LIMIT_MAP: dict = {
        "gpt-3.5-turbo": 50,
        "gpt-4": 5,
        "mistral": 2,
        "gemma": 2
    }
    # Mongo settings
    MONGO_COLLECTION_PROMPTS: str = "prompt_questions"
    MONGO_USERNAME: str
    MONGO_PASSWORD: str
    MONGO_HOST: str = "mongodb"
    MONGO_PORT: int = 27017
    MONGO_DBNAME: str = "benchmark"


settings = Settings()
