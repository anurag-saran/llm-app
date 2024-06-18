from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

from core import settings


api_key_header = APIKeyHeader(name=settings.API_KEY_NAME, auto_error=False)


async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == settings.API_KEY:
        return api_key
    else:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key",
        )
