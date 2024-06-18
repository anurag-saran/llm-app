from pydantic import BaseModel
from enum import Enum


class RequestModel(BaseModel):
    text: str


class StatusEnum(str, Enum): 
    PASSED = "passed"
    FAILED = "failed"


class ResponseModel(BaseModel):
    status: StatusEnum
    categories: list
