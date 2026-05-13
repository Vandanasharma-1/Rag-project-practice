"""
app/models/__init__.py

Makes `models` a Python package.
Exposes all Pydantic schemas for convenient importing.

Usage:
    from app.models import ChatRequest, ChatResponse
    # is equivalent to:
    from app.models.schemas import ChatRequest, ChatResponse
"""
from app.models.schemas import (
    UserRegisterRequest,
    UserLoginRequest,
    TokenResponse,
    UserResponse,
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentListItem,
    DocumentDeleteResponse,
    ChatRequest,
    ChatResponse,
    RetrievedChunk,
    HealthCheckResponse,
    ErrorResponse,
)

__all__ = [
    "UserRegisterRequest",
    "UserLoginRequest",
    "TokenResponse",
    "UserResponse",
    "DocumentUploadResponse",
    "DocumentListResponse",
    "DocumentListItem",
    "DocumentDeleteResponse",
    "ChatRequest",
    "ChatResponse",
    "RetrievedChunk",
    "HealthCheckResponse",
    "ErrorResponse",
]
