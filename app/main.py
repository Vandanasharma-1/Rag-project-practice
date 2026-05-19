"""

app/main.py — FastAPI Application Entry Point


WHY THIS FILE EXISTS:
    This is the HEART of the entire application.
    Every HTTP request starts and ends here.

    This file:
    1. Creates the FastAPI application instance
    2. Registers all middleware (CORS, request logging)
    3. Mounts all routers (auth, documents, chat)
    4. Defines startup/shutdown lifecycle events
    5. Exposes a health check endpoint
    6. Exposes a root endpoint with API info

HOW FastAPI WORKS (Beginner Explanation):
    FastAPI is a Python web framework. Think of it as a "traffic controller"
    for HTTP requests.

    When you type "http://localhost:8000/api/v1/chat/ask" in your browser
    or Postman, this is what happens:
    
    1. Your HTTP request arrives at the server (Uvicorn)
    2. Uvicorn passes it to FastAPI (this file)
    3. FastAPI runs all MIDDLEWARE (request logger, CORS headers)
    4. FastAPI finds the matching ROUTER (/api/v1/chat → chat_router)
    5. The router finds the matching ROUTE HANDLER (POST /ask)
    6. The route handler runs, produces a response
    7. Response travels back through middleware
    8. Response sent back to you

WHAT IS CORS?
    CORS = Cross-Origin Resource Sharing.
    
    Browsers block requests to DIFFERENT origins for security.
    "Origin" = scheme + domain + port.
    
    Your frontend: http://localhost:3000
    Your backend:  http://localhost:8000
    
    These are DIFFERENT origins! Without CORS headers on the backend,
    the browser will BLOCK the frontend from calling the API.
    
    Adding CORSMiddleware tells the browser: "It's OK, I allow requests
    from http://localhost:3000"

WHAT IS MIDDLEWARE (in order)?
    Middleware runs for EVERY request, in the order it's added.
    
    Request flow:
    → CORSMiddleware (adds CORS headers)
    → RequestLoggingMiddleware (logs the request)
    → Route Handler (your actual endpoint code)
    → RequestLoggingMiddleware (logs the response)
    → CORSMiddleware (nothing on way out)
    → Response sent

STARTUP EVENT:
    The @app.on_event("startup") function runs ONCE when the server
    starts. We use it to:
    - Initialize ChromaDB
    - Ensure directories exist
    - Load the embedding model (optionally)
    - Log that everything is ready

HOW IT CONNECTS:
    main.py → imports and includes all routers
    main.py → imports and adds all middleware
    main.py → called by run.py (which starts Uvicorn)

"""

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

#  Our Application Modules 
from app.config.config import settings
from app.utils.logger import logger, setup_logging
from app.middleware.request_logger import RequestLoggingMiddleware
from app.database.chroma_manager import chroma_manager

#  Routers 
# Each router handles a specific group of endpoints
from app.routers import auth_router, documents_router, chat_router



# STEP 1: CONFIGURE LOGGING (must be done before anything else)

# Set up Loguru logging before creating the FastAPI app.
# This ensures all startup messages are captured properly.
setup_logging(
    log_level=settings.log_level,
    log_dir=settings.log_dir,
    log_file=settings.log_file
)



# STEP 2: LIFESPAN CONTEXT MANAGER (Modern FastAPI startup/shutdown)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager — handles startup and shutdown.

    WHAT IS A CONTEXT MANAGER?
        A context manager is Python's way of managing resources.
        The `async with lifespan(app):` pattern means:
        - Code BEFORE `yield` runs at STARTUP
        - Code AFTER `yield` runs at SHUTDOWN

    WHY USE LIFESPAN INSTEAD OF @app.on_event?
        @app.on_event("startup") is the older approach.
        Lifespan is the MODERN, recommended approach in FastAPI.
        It's cleaner and handles async resources better.

    WHAT HAPPENS AT STARTUP:
        1. Ensure required directories exist (uploads, logs, chroma_db)
        2. Initialize ChromaDB connection
        3. Log that the app is ready

    WHAT HAPPENS AT SHUTDOWN:
        1. Log shutdown message
        2. Close any open connections (graceful shutdown)
    """

    # TARTUP    logger.info("-" * 60)
    logger.info(f"  {settings.app_name} v{settings.app_version}")
    logger.info("  Starting up...")
    logger.info("-" * 60)

    # Step 1: Ensure all required directories exist
    # This creates ./data/uploads, ./data/chroma_db, ./data/logs
    try:
        settings.ensure_directories()
        logger.info("Required directories verified")
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        raise

    # Step 2: Initialize ChromaDB vector database
    # This connects to (or creates) the persistent vector store
    try:
        chroma_manager.initialize()
        stats = chroma_manager.get_collection_stats()
        logger.info(
            f"ChromaDB initialized | "
            f"Collection: '{stats['name']}' | "
            f"Existing vectors: {stats['count']}"
        )
    except Exception as e:
        logger.error(f"✗ ChromaDB initialization failed: {e}")
        raise

    logger.info("=" * 60)
    logger.info("  Application ready! Listening for requests...")
    logger.info(f"  Docs: http://localhost:8000/docs")
    logger.info(f"  API:  http://localhost:8000/api/v1")
    logger.info("=" * 60)

    # IELD (App runs here)    yield  # ← The application runs while we wait here

    # HUTDOWN    logger.info("=" * 60)
    logger.info("  Shutting down gracefully...")
    logger.info("=" * 60)
    # Add any cleanup here: close DB connections, flush caches, etc.



# STEP 3: CREATE THE FASTAPI APPLICATION


app = FastAPI(
    # App metadata (shown in /docs Swagger UI)
    title=settings.app_name,
    description="""
## Enterprise GenAI Assistant with RAG Pipeline

A production-grade backend for document-based question answering using
**Retrieval-Augmented Generation (RAG)**.

### Features
-  **JWT Authentication** — Secure register/login
-  **Document Ingestion** — Upload PDF, TXT, DOCX files
-  **Semantic Search** — Find relevant content using vector similarity
-  **AI Answers** — Google Gemini generates contextual answers
- **Source Attribution** — Every answer shows its source chunks

### How to Use
1. **Register** a new account via `POST /api/v1/auth/register`
2. **Login** to get a JWT token via `POST /api/v1/auth/login`
3. **Authorize** by clicking the  button and entering `Bearer <your_token>`
4. **Upload** documents via `POST /api/v1/documents/upload`
5. **Ask** questions via `POST /api/v1/chat/ask`

### Tech Stack
- FastAPI + Uvicorn | Python async API
- ChromaDB | Vector database for embeddings
- SentenceTransformers | Text embedding model
- Google Gemini | LLM for answer generation
- JWT + bcrypt | Secure authentication
    """,
    version=settings.app_version,

    # OpenAPI documentation URLs
    docs_url="/docs",          # Swagger UI at /docs
    redoc_url="/redoc",        # ReDoc UI at /redoc
    openapi_url="/openapi.json",

    # Lifespan handles startup/shutdown
    lifespan=lifespan,

    # Contact and license info for API docs
    # contact={
    #     "name": "Enterprise RAG Team",
    #     "email": "support@enterprise-rag.com",
    # },
    # license_info={
    #     "name": "MIT License",
    # },
)



# STEP 4: ADD MIDDLEWARE

# IMPORTANT: Middleware is applied in REVERSE ORDER of addition.
# The LAST middleware added is the FIRST to process each request.
# So the request flow is:
# Request → RequestLoggingMiddleware → CORSMiddleware → Routes

#  Middleware 1: Request Logging 
# This logs every incoming request and outgoing response.
# Added FIRST so it wraps the entire request lifecycle.
app.add_middleware(RequestLoggingMiddleware)

#  Middleware 2: CORS (Cross-Origin Resource Sharing) 
# This adds CORS headers to ALL responses.
# Required for browser-based frontends to call this API.
app.add_middleware(
    CORSMiddleware,

    # allow_origins: Which frontend URLs can call this API
    # settings.cors_origins defaults to ["*"] (allow all)
    # In production, restrict to: ["https://yourapp.com"]
    allow_origins=settings.cors_origins,

    # allow_credentials: Allow cookies/auth headers in cross-origin requests
    allow_credentials=True,

    # allow_methods: Which HTTP methods are allowed
    # ["*"] means GET, POST, PUT, DELETE, PATCH, OPTIONS, HEAD
    allow_methods=["*"],

    # allow_headers: Which request headers are allowed
    # ["*"] includes Authorization (needed for JWT)
    allow_headers=["*"],
)



# STEP 5: REGISTER ROUTERS

# Each router handles a specific group of endpoints.
# The prefix defines the URL path prefix for all routes in that router.
# The tags are used to group endpoints in Swagger UI.

# Authentication router: /api/v1/auth/*
# Handles: register, login, logout, get-me
app.include_router(
    auth_router.router,
    prefix="/api/v1"
)

# Documents router: /api/v1/documents/*
# Handles: upload, list, get, delete documents
app.include_router(
    documents_router.router,
    prefix="/api/v1"
)

# Chat/RAG router: /api/v1/chat/*
# Handles: ask questions, semantic search, history
app.include_router(
    chat_router.router,
    prefix="/api/v1"
)



# STEP 6: CUSTOM EXCEPTION HANDLERS

# These catch specific error types and return clean JSON responses
# instead of raw Python exceptions.

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Handle all HTTP exceptions (404, 401, 403, 500, etc.).
    
    WHY CUSTOM HANDLER?
        FastAPI's default error response format varies.
        Our custom handler ensures EVERY error looks the same:
        {
            "error": "NOT_FOUND",
            "message": "...",
            "status_code": 404
        }
    
    This makes frontend error handling much simpler.
    """
    # Map status codes to error names
    error_names = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        409: "CONFLICT",
        413: "PAYLOAD_TOO_LARGE",
        415: "UNSUPPORTED_MEDIA_TYPE",
        422: "UNPROCESSABLE_ENTITY",
        429: "TOO_MANY_REQUESTS",
        500: "INTERNAL_SERVER_ERROR",
        503: "SERVICE_UNAVAILABLE",
    }

    error_name = error_names.get(exc.status_code, "HTTP_ERROR")

    logger.warning(f"HTTP {exc.status_code} error: {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": error_name,
            "message": str(exc.detail),
            "status_code": exc.status_code,
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle Pydantic validation errors (invalid request body).
    
    WHEN DOES THIS TRIGGER?
        When the client sends data that doesn't match the schema.
        Example: Sending {"email": "not-an-email"} for a field
        expecting EmailStr — Pydantic raises RequestValidationError.
    
    We format the errors to be more readable.
    """
    # Extract human-readable error messages
    errors = []
    for error in exc.errors():
        field = " → ".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"],
        })

    logger.warning(f"Validation error on {request.url}: {errors}")

    return JSONResponse(
        status_code=422,
        content={
            "error": "VALIDATION_ERROR",
            "message": "Request validation failed. Check the 'details' field for specific errors.",
            "status_code": 422,
            "details": errors,
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Catch-all handler for unexpected exceptions.
    
    WHY?
        Without this, unhandled Python exceptions return a raw 500 error
        that might expose internal code or stack traces.
        
        We catch everything, log it, and return a clean response.
    """
    logger.error(f"Unhandled exception on {request.method} {request.url}: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred. Please try again or contact support.",
            "status_code": 500,
        }
    )



# STEP 7: DEFINE CORE ENDPOINTS


@app.get(
    "/",
    tags=["System"],
    summary="Root endpoint",
    description="Returns basic API information and available endpoints."
)
async def root():
    """
    Root endpoint — returns API info and quick-start links.
    
    This is what you see when you visit http://localhost:8000/
    
    Returns:
        Dict with app info, version, and useful links
    """
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": "Enterprise GenAI Assistant with RAG Pipeline",
        "links": {
            "swagger_docs": "http://localhost:8000/docs",
            "redoc_docs":   "http://localhost:8000/redoc",
            "health_check": "http://localhost:8000/health",
            "openapi_spec": "http://localhost:8000/openapi.json",
        },
        "quick_start": {
            "step_1": "POST /api/v1/auth/register — Create account",
            "step_2": "POST /api/v1/auth/login — Get JWT token",
            "step_3": "POST /api/v1/documents/upload — Upload a document",
            "step_4": "POST /api/v1/chat/ask — Ask a question",
        },
        "api_prefix": "/api/v1",
    }


@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    description="""
    Health check endpoint for monitoring systems.
    
    Returns the status of all service components:
    - **vector_db**: ChromaDB connection status
    - **llm**: Google Gemini API availability
    - **embedding_model**: SentenceTransformers model status
    
    Used by load balancers, Kubernetes, and monitoring tools
    to determine if this instance is healthy.
    """
)
async def health_check():
    """
    Comprehensive health check for all system components.
    
    WHY HEALTH CHECKS MATTER:
        In production, your app runs on multiple servers.
        A load balancer sends traffic to healthy servers.
        If this endpoint returns non-200, the server is removed
        from the pool. When it recovers, it's added back.
        
        This is "automatic recovery" — no human intervention needed.
    
    Returns:
        Health status dict with component-level details
    """
    start_time = time.time()

    # Check ChromaDB
    vector_db_healthy = chroma_manager.health_check()

    # Check LLM (Gemini) — only do a quick check, not a full API call
    try:
        from app.services.llm_client import llm_client
        llm_healthy = llm_client.health_check()
    except Exception:
        llm_healthy = False

    # Check embedding model by attempting to load it
    try:
        from app.services.vector_store import vector_store
        # Just access the model — if it loads, it's healthy
        _ = vector_store._get_embedding_model()
        embedding_healthy = True
    except Exception:
        embedding_healthy = False

    # Overall health: all critical services must be healthy
    # LLM is not critical for health check (it's an external API)
    all_critical_healthy = vector_db_healthy

    overall_status = "healthy" if all_critical_healthy else "degraded"

    # Get vector DB stats
    db_stats = chroma_manager.get_collection_stats() if vector_db_healthy else {}

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    response = {
        "status": overall_status,
        "version": settings.app_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "response_time_ms": elapsed_ms,
        "services": {
            "vector_db": {
                "status": "connected" if vector_db_healthy else "disconnected",
                "collection": settings.chroma_collection_name,
                "stored_vectors": db_stats.get("count", 0),
            },
            "llm": {
                "status": "available" if llm_healthy else "unavailable",
                "model": settings.gemini_model,
                "provider": "Google Gemini",
            },
            "embedding_model": {
                "status": "loaded" if embedding_healthy else "unloaded",
                "model": settings.embedding_model,
                "dimensions": settings.embedding_dimension,
            },
        },
        "configuration": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k_default": settings.top_k,
            "max_file_size_mb": settings.max_file_size // (1024 * 1024),
            "allowed_file_types": settings.allowed_extensions,
        }
    }

    # Return 503 if unhealthy (important for load balancers!)
    # A 200 response means "I'm healthy, send me traffic"
    # A 503 response means "I'm sick, don't send me traffic"
    if overall_status != "healthy":
        return JSONResponse(status_code=503, content=response)

    return response


@app.get(
    "/api/v1",
    tags=["System"],
    summary="API v1 root",
    description="Returns available API endpoints organized by category."
)
async def api_root():
    """
    API v1 root — lists all available endpoints.
    
    A helpful directory of all API routes for developers
    integrating with this service.
    """
    return {
        "version": "v1",
        "base_url": "/api/v1",
        "endpoints": {
            "authentication": {
                "register": "POST /api/v1/auth/register",
                "login":    "POST /api/v1/auth/login",
                "logout":   "POST /api/v1/auth/logout",
                "me":       "GET  /api/v1/auth/me",
            },
            "documents": {
                "upload":   "POST   /api/v1/documents/upload",
                "list":     "GET    /api/v1/documents/",
                "get":      "GET    /api/v1/documents/{document_id}",
                "delete":   "DELETE /api/v1/documents/{document_id}",
                "stats":    "GET    /api/v1/documents/stats/overview",
            },
            "chat": {
                "ask":      "POST   /api/v1/chat/ask",
                "search":   "POST   /api/v1/chat/search",
                "history":  "GET    /api/v1/chat/history",
                "clear":    "DELETE /api/v1/chat/history",
            },
        },
        "documentation": {
            "swagger":  "/docs",
            "redoc":    "/redoc",
            "openapi":  "/openapi.json",
        }
    }
