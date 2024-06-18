from fastapi import APIRouter

from api import inference


api_router = APIRouter()

api_router.include_router(inference.router, prefix="/inference")
