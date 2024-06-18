from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from auth import get_api_key
from core.logger import logger
from schemas import RequestModel, ResponseModel
from services.detection_service import Hallucinations


router = APIRouter(tags=["inference"])

hallucinations_detector = Hallucinations()


@router.post("/hallucinations", response_model=ResponseModel)
async def post_inference(request: RequestModel, api_key: str = Depends(get_api_key)):
    try:
        logger.debug(f"Request body: {request.llm_prompt}")
        logger.debug(f"Request body: {request.llm_response}")
        status = await hallucinations_detector.scan(llm_prompt=request.llm_prompt, llm_response=request.llm_response)

        response = ResponseModel(
            status=status,
        )
        return response

    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error", "details": str(e)},
        )
