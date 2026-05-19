"""

app/models/schemas.py — Pydantic Data Models (Schemas)


WHY THIS FILE EXISTS:
    This file defines the "shapes" of data that flows in and out
    of our API. These are like contracts:
    - REQUEST schemas: What data must the client send?
    - RESPONSE schemas: What data does the server return?

    Pydantic validates every request/response automatically.
    If a field is missing or wrong type, it returns a clear error.

WHAT IS PYDANTIC?
    Pydantic is a data validation library. You define a class
    with type annotations, and Pydantic enforces them at runtime.

    Example:
        class User(BaseModel):
            email: str
            age: int

        User(email="john@example.com", age=25)  ✓ Valid
        User(email="john@example.com", age="not-a-number")  ✗ Error!

    In FastAPI, schemas serve as:
    1. Request body parsing (validates incoming JSON)
    2. Response serialization (formats outgoing JSON)
    3. API documentation (auto-generates OpenAPI/Swagger docs)

SCHEMA NAMING CONVENTION:
    - [Name]Request  → Data the client sends TO the server
    - [Name]Response → Data the server sends BACK to the client
    - [Name]Create   → Data needed to create a new resource
    - [Name]Base     → Base class with common fields

HOW IT CONNECTS:
    schemas.py → used by ALL routers (auth, documents, chat)
    schemas.py → used by ALL services
    FastAPI uses these to auto-generate /docs Swagger UI

"""

from typing import Optional, List, Any
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, field_validator



# AUTHENTICATION SCHEMAS


class UserRegisterRequest(BaseModel):
    """
    Schema for user registration.

    WHAT IS EmailStr?
        A Pydantic type that validates email format.
        "not-an-email" → validation error
        "user@example.com" → accepted ✓

    Example request body:
        {
            "email": "john@example.com",
            "password": "SecurePass123",
            "full_name": "John Doe"
        }
    """
    email: EmailStr = Field(..., description="User's email address", example="john@example.com")
    password: str = Field(..., min_length=8, description="Password (min 8 characters)", example="SecurePass123")
    full_name: str = Field(..., min_length=2, max_length=100, description="User's full name", example="John Doe")

    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """
        Ensure password meets minimum strength requirements.

        WHY?
            Weak passwords like "12345678" are a security risk.
            We enforce at least one letter and one number.
        """
        if not any(c.isalpha() for c in v):
            raise ValueError("Password must contain at least one letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserLoginRequest(BaseModel):
    """
    Schema for user login.

    Example request body:
        {
            "email": "john@example.com",
            "password": "SecurePass123"
        }
    """
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., description="User's password")


class TokenResponse(BaseModel):
    """
    Schema for the JWT token returned after successful login.

    Example response:
        {
            "access_token": "eyJhbGciOiJIUzI1NiJ9...",
            "token_type": "bearer",
            "expires_in": 3600,
            "user_email": "john@example.com"
        }

    HOW TOKEN_TYPE WORKS:
        The client sends the token in the Authorization header:
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9..."
        The "Bearer" prefix tells the server it's a JWT token.
    """
    access_token: str = Field(..., description="The JWT access token")
    token_type: str = Field(default="bearer", description="Token type (always 'bearer')")
    expires_in: int = Field(..., description="Token validity in seconds")
    user_email: str = Field(..., description="Email of the authenticated user")


class UserResponse(BaseModel):
    """
    Schema for returning user info (without password).

    We NEVER return the password, even as a hash.
    This schema intentionally excludes it.
    """
    email: str
    full_name: str
    created_at: datetime
    is_active: bool = True



# DOCUMENT SCHEMAS


class DocumentUploadResponse(BaseModel):
    """
    Response after successfully uploading a document.

    Example response:
        {
            "document_id": "a47f3d2c-8b9e-4f1a",
            "filename": "annual_report.pdf",
            "status": "processed",
            "chunks_created": 42,
            "file_size": "2.3 MB",
            "message": "Document processed successfully"
        }
    """
    document_id: str = Field(..., description="Unique ID for this document")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Processing status: 'processed' or 'failed'")
    chunks_created: int = Field(..., description="Number of text chunks created from this document")
    file_size: str = Field(..., description="Human-readable file size")
    message: str = Field(..., description="Success or error message")


class DocumentListItem(BaseModel):
    """
    A single item in the list of uploaded documents.
    """
    document_id: str
    filename: str
    chunks_count: int
    upload_date: str
    file_type: str


class DocumentListResponse(BaseModel):
    """
    Response for listing all uploaded documents.

    Example response:
        {
            "documents": [...],
            "total": 5
        }
    """
    documents: List[DocumentListItem]
    total: int = Field(..., description="Total number of documents")


class DocumentDeleteResponse(BaseModel):
    """Response after deleting a document."""
    document_id: str
    message: str
    chunks_deleted: int



# CHAT / RAG SCHEMAS


class ChatRequest(BaseModel):
    """
    Schema for asking a question to the RAG assistant.

    Example request body:
        {
            "question": "What was the revenue in Q3 2023?",
            "top_k": 5
        }

    top_k: How many relevant document chunks to retrieve.
    More chunks = more context = potentially better answers,
    but also more tokens and slower response.
    """
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The question to ask the AI assistant",
        example="What was the company's revenue in Q3 2023?"
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of relevant chunks to retrieve (1-20)"
    )


class RetrievedChunk(BaseModel):
    """
    A single retrieved document chunk with its relevance score.

    WHAT IS A RELEVANCE SCORE?
        When we search the vector database for chunks similar
        to the user's question, each result gets a score.

        Score range: 0.0 to 1.0
        - 1.0 = Perfect match (very rare)
        - 0.8+ = High relevance
        - 0.5-0.8 = Medium relevance
        - <0.5 = Low relevance

    WHY INCLUDE CHUNKS IN THE RESPONSE?
        Transparency. Users should know WHERE the AI's answer
        came from. This is "source attribution" — a key feature
        of trustworthy RAG systems.
    """
    chunk_id: str = Field(..., description="Unique ID of this chunk")
    content: str = Field(..., description="The actual text of this chunk")
    relevance_score: float = Field(..., description="Similarity score (0-1)")
    document_id: str = Field(..., description="ID of the source document")
    filename: str = Field(..., description="Name of the source document file")
    chunk_index: int = Field(..., description="Position of this chunk in the document")


class ChatResponse(BaseModel):
    """
    Complete response from the RAG pipeline.

    This is the most important schema — it shows:
    1. The user's original question
    2. The AI-generated answer
    3. The source chunks used to generate the answer
    4. Performance metrics

    Example response:
        {
            "question": "What was revenue in Q3?",
            "answer": "Based on the annual report, revenue in Q3 was $4.2B...",
            "retrieved_chunks": [
                {
                    "chunk_id": "abc123",
                    "content": "Q3 2023 revenue reached $4.2 billion...",
                    "relevance_score": 0.92,
                    "document_id": "doc_001",
                    "filename": "annual_report.pdf",
                    "chunk_index": 15
                }
            ],
            "processing_time_seconds": 1.234,
            "model_used": "gemini-1.5-flash",
            "chunks_retrieved": 5,
            "status": "success"
        }
    """
    question: str = Field(..., description="The user's original question")
    answer: str = Field(..., description="AI-generated answer based on retrieved context")
    retrieved_chunks: List[RetrievedChunk] = Field(
        ...,
        description="The document chunks used to generate the answer"
    )
    processing_time_seconds: float = Field(..., description="Time taken to process the request")
    model_used: str = Field(..., description="The LLM model that generated the answer")
    chunks_retrieved: int = Field(..., description="Number of chunks retrieved from the database")
    status: str = Field(default="success", description="Request status: 'success' or 'error'")
    error_message: Optional[str] = Field(default=None, description="Error details if status is 'error'")



# HEALTH CHECK SCHEMA


class HealthCheckResponse(BaseModel):
    """
    Response for the /health endpoint.

    WHY HEALTH CHECK?
        In production, monitoring systems (like Kubernetes, AWS ELB)
        periodically call /health to check if the service is alive.
        If it returns 200 OK, the service is healthy.
        If it fails, the monitoring system can restart/alert.

    Example response:
        {
            "status": "healthy",
            "version": "1.0.0",
            "services": {
                "vector_db": "connected",
                "llm": "available"
            }
        }
    """
    status: str = Field(..., description="Overall health: 'healthy' or 'unhealthy'")
    version: str = Field(..., description="Application version")
    timestamp: str = Field(..., description="Current timestamp")
    services: dict = Field(..., description="Status of each service component")



# ERROR SCHEMA


class ErrorResponse(BaseModel):
    """
    Standard error response format.

    WHY STANDARDIZE ERRORS?
        Without a standard format, different errors look different:
        - Some return {"error": "message"}
        - Others return {"detail": "message"}
        - Others return raw strings

        A standard format makes error handling easier for the frontend.

    Example response:
        {
            "error": "DOCUMENT_NOT_FOUND",
            "message": "Document with ID 'abc123' not found",
            "status_code": 404
        }
    """
    error: str = Field(..., description="Error code/type")
    message: str = Field(..., description="Human-readable error message")
    status_code: int = Field(..., description="HTTP status code")
    details: Optional[Any] = Field(default=None, description="Additional error details")
