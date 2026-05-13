"""
app/database/__init__.py

Makes `database` a Python package.
Exposes the ChromaDB manager singleton.

WHAT IS THE DATABASE LAYER?
    The database layer handles all data persistence operations.
    In our case, the "database" is ChromaDB — a vector database
    that stores text embeddings (numerical representations of text).

    Unlike SQL databases (tables, rows, columns), ChromaDB stores:
    - Vectors: Lists of floats representing text meaning
    - Documents: The original text of each chunk
    - Metadata: Extra info like filename, chunk index, etc.

Usage:
    from app.database import chroma_manager
    chroma_manager.initialize()
    collection = chroma_manager.collection
"""
from app.database.chroma_manager import chroma_manager

__all__ = ["chroma_manager"]
