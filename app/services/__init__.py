"""
app/services/__init__.py

Makes `services` a Python package.
Exposes all service singleton instances.

WHAT IS A SERVICE LAYER?
    Services contain the BUSINESS LOGIC of the application.
    They sit between the API routers and the data layer.

    Routers handle HTTP (request/response)
    Services handle LOGIC (what to do with the data)
    Database handles STORAGE (where data lives)

    This separation makes code:
    - Easier to test (test services without HTTP)
    - Easier to reuse (multiple routers can use the same service)
    - Easier to maintain (change logic in one place)

Usage:
    from app.services import document_processor, vector_store, llm_client, rag_pipeline
"""
from app.services.document_processor import document_processor
from app.services.vector_store import vector_store
from app.services.llm_client import llm_client
from app.services.rag_pipeline import rag_pipeline

__all__ = [
    "document_processor",
    "vector_store",
    "llm_client",
    "rag_pipeline",
]
