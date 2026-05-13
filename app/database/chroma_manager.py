"""
==============================================================
app/database/chroma_manager.py — ChromaDB Vector Database Manager
==============================================================

WHY THIS FILE EXISTS:
    This file manages our vector database (ChromaDB).

    A vector database is like a regular database BUT instead of
    storing data in tables with rows and columns, it stores
    "embeddings" — numerical vectors that represent text meaning.

WHAT ARE EMBEDDINGS AND VECTORS?
    Imagine every piece of text can be converted to a list of numbers.

    "The company revenue grew in Q3" → [0.23, -0.11, 0.87, 0.45, ...]
    "Sales increased in the third quarter" → [0.21, -0.09, 0.85, 0.47, ...]

    Notice these two SIMILAR sentences produce SIMILAR vectors!
    (The numbers are close to each other)

    "The cat sat on the mat" → [0.92, 0.73, -0.44, 0.11, ...]

    This DIFFERENT sentence produces DIFFERENT numbers.

    The vector database uses this property to find similar text.
    When you ask "What was revenue in Q3?", it converts your question
    to a vector and finds document chunks with the CLOSEST vectors.
    This is "semantic search" — searching by MEANING, not exact words.

WHAT IS CHROMADB?
    ChromaDB is an open-source vector database that:
    - Runs locally (no cloud service needed!)
    - Stores vectors and their metadata on disk
    - Performs fast similarity searches
    - Perfect for development and small-medium production use

HOW SIMILARITY SEARCH WORKS:
    1. "Q3 revenue?" → embedding → [0.23, -0.11, 0.87, ...]
    2. Calculate distance between query vector and ALL stored vectors
    3. Return the N closest vectors (these are the relevant chunks)
    4. "Distance" = how semantically different two texts are

    Mathematical method: Cosine Similarity
    - Cosine similarity of 1.0 = identical meaning
    - Cosine similarity of 0.0 = completely different

HOW IT CONNECTS:
    chroma_manager.py → used by vector_store.py (add/search chunks)
    vector_store.py → used by rag_pipeline.py (the full RAG flow)
==============================================================
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import Optional

from app.config.config import settings
from app.utils.logger import logger


class ChromaManager:
    """
    Manages the ChromaDB client and collection lifecycle.

    DESIGN PATTERN: Singleton
        We create ONE ChromaManager instance and reuse it.
        Creating multiple database connections is wasteful.

    WHAT IS A "COLLECTION"?
        In ChromaDB, a "collection" is like a table in SQL.
        It groups related vectors together.
        We have one collection: "enterprise_docs" for all document chunks.

    WHAT CHROMADB STORES FOR EACH CHUNK:
        - id:         Unique identifier for this chunk
        - document:   The actual text of the chunk
        - embedding:  The numerical vector [0.23, -0.11, 0.87, ...]
        - metadata:   Extra info like {"filename": "report.pdf", "page": 3}
    """

    def __init__(self):
        """
        Initialize ChromaDB client and collection.

        PersistentClient vs EphemeralClient:
        - PersistentClient: Saves data to DISK. Data survives restarts. ✓
        - EphemeralClient: Stores in MEMORY. Data lost on restart. ✗ (for our use)

        We use PersistentClient so uploaded documents persist between
        server restarts.
        """
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None
        self._initialized = False

    def initialize(self) -> None:
        """
        Connect to ChromaDB and get/create the collection.

        WHY SEPARATE INITIALIZATION?
            We don't connect in __init__ because the app might not
            be ready yet when the object is created. We call this
            explicitly during the FastAPI startup event.

        WHAT DOES "get_or_create_collection" DO?
            - If collection exists: returns it (loads existing data)
            - If not: creates a new empty collection
            This is idempotent — safe to call multiple times.
        """
        try:
            logger.info(f"Initializing ChromaDB at path: {settings.chroma_db_path}")

            # Create a persistent ChromaDB client
            # All data is saved to settings.chroma_db_path
            self._client = chromadb.PersistentClient(
                path=settings.chroma_db_path,
                settings=ChromaSettings(
                    anonymized_telemetry=False,  # Don't send usage data to ChromaDB
                    allow_reset=True             # Allow resetting the database in tests
                )
            )

            # Get or create our document collection
            # This collection stores all document chunk embeddings
            self._collection = self._client.get_or_create_collection(
                name=settings.chroma_collection_name,
                # Metadata about the collection
                metadata={
                    "description": "Enterprise document embeddings for RAG",
                    "embedding_model": settings.embedding_model,
                    # cosine distance is best for semantic similarity
                    # options: "cosine", "l2", "ip"
                    "hnsw:space": "cosine"
                }
            )

            self._initialized = True
            count = self._collection.count()
            logger.info(
                f"ChromaDB initialized successfully | "
                f"Collection: '{settings.chroma_collection_name}' | "
                f"Existing vectors: {count}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    @property
    def collection(self):
        """
        Property to access the ChromaDB collection.

        WHY A PROPERTY WITH GUARD?
            If someone calls this before initialize() runs,
            we raise a clear error instead of a confusing AttributeError.
        """
        if not self._initialized or self._collection is None:
            raise RuntimeError(
                "ChromaDB not initialized. Call initialize() first. "
                "This usually happens if the startup event didn't run."
            )
        return self._collection

    @property
    def client(self):
        """Property to access the raw ChromaDB client."""
        if not self._initialized or self._client is None:
            raise RuntimeError("ChromaDB not initialized. Call initialize() first.")
        return self._client

    def get_collection_stats(self) -> dict:
        """
        Get statistics about the current collection.

        Returns:
            Dict with count of stored vectors and collection metadata

        Example return:
            {
                "name": "enterprise_docs",
                "count": 247,
                "metadata": {"embedding_model": "all-MiniLM-L6-v2"}
            }
        """
        if not self._initialized:
            return {"status": "not_initialized"}

        return {
            "name": settings.chroma_collection_name,
            "count": self._collection.count(),
            "metadata": self._collection.metadata,
        }

    def reset_collection(self) -> None:
        """
        Delete and recreate the collection (DESTRUCTIVE!).

        WARNING: This deletes ALL stored embeddings!
        Only use this for testing or if you want to start fresh.
        """
        if not self._initialized:
            return

        logger.warning(f"Resetting ChromaDB collection: {settings.chroma_collection_name}")
        self._client.delete_collection(settings.chroma_collection_name)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Collection reset complete")

    def health_check(self) -> bool:
        """
        Check if ChromaDB is responsive.

        Returns:
            True if healthy, False otherwise
        """
        try:
            if not self._initialized:
                return False
            # A simple count query to test connectivity
            self._collection.count()
            return True
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return False


# ==============================================================
# SINGLETON INSTANCE
# ==============================================================

# Create a single ChromaDB manager instance for the whole application.
# This is imported by vector_store.py.
# We call chroma_manager.initialize() in the FastAPI startup event.
chroma_manager = ChromaManager()
