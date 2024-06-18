from pydantic import BaseModel
from enum import Enum


class RequestModel(BaseModel):
    text_a: str
    text_b: str
    metadata: dict


class StatusEnum(str, Enum): 
    PASSED = "passed"
    FAILED = "failed"


class ResponseModel(BaseModel):
    status: StatusEnum
    score: float
