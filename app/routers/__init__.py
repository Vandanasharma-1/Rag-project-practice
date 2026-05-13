"""
app/routers/__init__.py

Makes `routers` a Python package.
Exposes all router modules for convenient importing in main.py.

WHAT IS A ROUTER?
    In FastAPI, an APIRouter groups related endpoints together.
    Instead of defining all 15+ routes in main.py, we split them:
    - auth_router   → /api/v1/auth/* endpoints
    - documents_router → /api/v1/documents/* endpoints
    - chat_router   → /api/v1/chat/* endpoints

Usage in main.py:
    from app.routers import auth_router, documents_router, chat_router
    app.include_router(auth_router.router, prefix="/api/v1")
"""
from app.routers import auth_router, documents_router, chat_router

__all__ = ["auth_router", "documents_router", "chat_router"]
