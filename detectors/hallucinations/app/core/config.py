import dotenv
from pydantic_settings import BaseSettings

dotenv.load_dotenv()


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_TEMPERATURE: float = 1
    OPENAI_MODEL_NAME: str = "gpt-3.5-turbo"
    OPENAI_MAX_TOKENS: int = 20

    # Detector user
    API_KEY: str = "actual_api_key_123"
    API_KEY_NAME: str = "access_token"


settings = Settings()
