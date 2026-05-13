"""
==============================================================
app/config/config.py — Application Configuration
==============================================================

WHY THIS FILE EXISTS:
    This is the SINGLE SOURCE OF TRUTH for all configuration.
    Instead of hardcoding values like API keys or file paths
    scattered across 20 files, we define them ONCE here.

    If you need to change the database path, you change it
    in ONE place — here — and the whole app picks it up.

HOW IT WORKS:
    1. Pydantic's BaseSettings reads from environment variables.
    2. It also reads from the .env file automatically.
    3. Every other file imports `settings` from here.
    4. Type validation ensures values are the right type.

BEGINNER TIP:
    Think of this file as the "settings panel" for the entire
    application. All knobs and dials are here.

HOW FILES CONNECT:
    .env file → (loaded by dotenv) → config.py → all other files
    Any file that needs a setting does: from app.config.config import settings
==============================================================
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application Settings using Pydantic BaseSettings.

    WHAT IS BaseSettings?
        It's a special Pydantic class that automatically reads
        values from environment variables AND from a .env file.

        So if you set GEMINI_API_KEY=abc123 in your .env file,
        this class automatically sets self.gemini_api_key = "abc123"

    WHAT IS lru_cache (below)?
        It ensures Settings() is only created ONCE. Every time
        get_settings() is called, it returns the SAME object.
        This is a performance optimization called "singleton pattern".
    """

    # --- App Metadata ---
    # Basic information about this application
    app_name: str = Field(default="Enterprise RAG Assistant", description="Name of the application")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")

    # --- Google Gemini API ---
    # This key is required to call Google's Gemini LLM
    # Get yours at: https://makersuite.google.com/app/apikey
    gemini_api_key: str = Field(..., description="Google Gemini API key (required)")

    # Which Gemini model to use for generating answers
    # gemini-1.5-flash is free tier and fast
    gemini_model: str = Field(default="gemini-1.5-flash", description="Gemini model name")

    # --- JWT Authentication Settings ---
    # JWT = JSON Web Token — used to authenticate users after login
    jwt_secret: str = Field(..., description="Secret key for JWT signing (required)")

    # Hashing algorithm — HS256 is the standard for JWT
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")

    # How long (in minutes) before a login token expires
    access_token_expire_minutes: int = Field(default=60, description="Token expiry in minutes")

    # --- ChromaDB Vector Database ---
    # ChromaDB stores our document embeddings (numerical vectors)
    chroma_db_path: str = Field(default="./data/chroma_db", description="ChromaDB storage path")

    # Collection name inside ChromaDB where vectors are stored
    chroma_collection_name: str = Field(default="enterprise_docs", description="ChromaDB collection name")

    # --- File Upload Settings ---
    # Where uploaded documents are stored before processing
    upload_dir: str = Field(default="./data/uploads", description="Document upload directory")

    # Maximum file size: 50MB (50 * 1024 * 1024 bytes)
    max_file_size: int = Field(default=50 * 1024 * 1024, description="Max upload size in bytes")

    # Allowed document types for upload
    allowed_extensions: list[str] = Field(
        default=[".pdf", ".txt", ".docx"],
        description="Allowed file extensions"
    )

    # --- Logging Settings ---
    log_level: str = Field(default="INFO", description="Logging verbosity level")
    log_dir: str = Field(default="./data/logs", description="Log files directory")
    log_file: str = Field(default="app.log", description="Log filename")

    # --- RAG Pipeline Settings ---
    # These control how documents are split into chunks

    # chunk_size: How many characters per chunk
    # If a document has 10,000 chars and chunk_size=500, we get ~20 chunks
    chunk_size: int = Field(default=500, description="Characters per text chunk")

    # chunk_overlap: How many characters overlap between adjacent chunks
    # Overlap ensures context isn't lost at chunk boundaries
    chunk_overlap: int = Field(default=50, description="Character overlap between chunks")

    # top_k: How many relevant chunks to retrieve per question
    # Higher = more context but slower and more tokens
    top_k: int = Field(default=5, description="Number of chunks to retrieve")

    # --- Embedding Model ---
    # SentenceTransformers model for converting text → vectors
    # all-MiniLM-L6-v2 is small, fast, and free — perfect for learning
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformers embedding model"
    )

    # Embedding dimension (all-MiniLM-L6-v2 produces 384-dim vectors)
    embedding_dimension: int = Field(default=384, description="Embedding vector dimensions")

    # --- CORS Settings ---
    # CORS = Cross-Origin Resource Sharing
    # Controls which frontend URLs can call our backend API
    # "*" means allow ALL origins (fine for development, restrict in production)
    cors_origins: list[str] = Field(default=["*"], description="Allowed CORS origins")

    class Config:
        """
        Pydantic config class.

        env_file: Tells Pydantic to read from .env file
        case_sensitive: Environment variables are case-insensitive
        extra: Allow extra fields without error
        """
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"

    def ensure_directories(self) -> None:
        """
        Create required directories if they don't exist.

        WHY?
            Our app needs certain folders (uploads, logs, chroma_db).
            If they don't exist, file operations will fail with errors.
            This method is called at startup to ensure everything exists.
        """
        dirs_to_create = [
            self.upload_dir,
            self.chroma_db_path,
            self.log_dir,
        ]

        for directory in dirs_to_create:
            Path(directory).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached singleton instance of Settings.

    WHY lru_cache?
        Without it, every function call creates a NEW Settings object,
        reading from .env every time — slow and wasteful.

        With lru_cache, the FIRST call creates the object.
        All subsequent calls return the CACHED version instantly.

        This is the "singleton pattern" — only ONE instance exists.

    USAGE:
        from app.config.config import get_settings, settings
        settings = get_settings()
        print(settings.gemini_api_key)
    """
    return Settings()


# Convenience: a module-level `settings` object
# Import this directly: from app.config.config import settings
settings = get_settings()
