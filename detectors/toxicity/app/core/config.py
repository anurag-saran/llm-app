import dotenv
from pydantic_settings import BaseSettings

dotenv.load_dotenv()


class Settings(BaseSettings):
    # Detector user
    API_KEY: str = "actual_api_key_123"
    API_KEY_NAME: str = "access_token"

    THRESHOLD: float = 0.6


settings = Settings()
