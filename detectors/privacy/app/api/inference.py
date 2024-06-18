from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from core.logger import logger
from schemas import RequestModel, ResponseModel
from services.detection_service import PrivacyChecker


from auth import get_api_key

router = APIRouter(tags=["inference"])


@router.post("/privacy", response_model=ResponseModel)
async def post_inference(request: RequestModel, api_key: str = Depends(get_api_key)):
    try:
        logger.debug(f"POST /inference")
        detector = PrivacyChecker()
        return detector.scan(request.text)

    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error", "details": str(e)},
        )
