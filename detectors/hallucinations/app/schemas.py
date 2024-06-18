from pydantic import BaseModel
from enum import Enum


class RequestModel(BaseModel):
    llm_prompt: str
    llm_response: str


class StatusEnum(str, Enum): 
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ResponseModel(BaseModel):
    status: StatusEnum
