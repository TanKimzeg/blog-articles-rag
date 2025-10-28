from typing import Generic, Optional, TypeVar
from pydantic import BaseModel, Field
import time

T = TypeVar("T")

class ApiResponse(BaseModel, Generic[T]):
    success: bool = True
    code: int = 0
    message: str
    data: Optional[T] = None
    ts: int = Field(default_factory=lambda: int(time.time() * 1000))

def ok(data: T = None, message: str = "ok") -> ApiResponse[T]:
    return ApiResponse(success=True, code=0, message=message, data=data)

def fail(code: int = 1, message: str = "error", data: T = None) -> ApiResponse[T]:
    return ApiResponse(success=False, code=code, message=message, data=data)