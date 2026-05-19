"""

app/routers/documents_router.py — Document Management Routes


WHY THIS FILE EXISTS:
    This file handles all document-related API endpoints:
    - POST /documents/upload  → Upload and process a document
    - GET  /documents/        → List all uploaded documents
    - GET  /documents/{id}    → Get a specific document's info
    - DELETE /documents/{id}  → Delete a document

    These endpoints are the "data ingestion" pipeline of the RAG system.
    Without documents, the chat system has nothing to answer from.

DOCUMENT UPLOAD FLOW:
    1. User sends HTTP POST with file attached
    2. FastAPI receives the file as UploadFile
    3. We validate: file type, file size
    4. We read the file bytes into memory
    5. document_processor extracts text from the file
    6. Text is split into chunks
    7. vector_store generates embeddings for each chunk
    8. Embeddings stored in ChromaDB
    9. Return success response with stats

WHAT IS UploadFile?
    FastAPI's special type for handling file uploads.
    It gives access to:
    - file.filename: Original filename
    - file.content_type: MIME type (e.g., "application/pdf")
    - file.read(): Read file bytes

HOW IT CONNECTS:
    documents_router.py ← called by HTTP clients (frontend, curl, Postman)
    documents_router.py → uses document_processor.py (extract text)
    documents_router.py → uses vector_store.py (store embeddings)
    documents_router.py → uses auth_router.py (get_current_user dependency)

"""

import time
from typing import List, Optional

from fastapi import (
    APIRouter, Depends, HTTPException,
    UploadFile, File, status, Query
)

from app.models.schemas import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentListItem,
    DocumentDeleteResponse,
    ErrorResponse
)
from app.services.document_processor import document_processor
from app.services.vector_store import vector_store
from app.database.chroma_manager import chroma_manager
from app.routers.auth_router import get_current_user
from app.config.config import settings
from app.utils.helpers import (
    generate_unique_id,
    create_unique_filename,
    is_allowed_file,
    format_file_size
)
from app.utils.logger import logger


# ROUTER SETUP


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized — JWT token required"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported file type"},
    }
)


# IN-MEMORY DOCUMENT REGISTRY

# ⚠️ DEMO ONLY: In production, use a database!
# Stores metadata about uploaded documents
# {document_id: {filename, chunks, date, user_email, ...}}
documents_registry: dict = {}



# API ROUTES


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and process a document",
    description="""
    Upload a document (PDF, TXT, or DOCX) to be processed and indexed for RAG.

    **Process:**
    1. File is validated (type and size)
    2. Text is extracted from the document
    3. Text is split into overlapping chunks
    4. Chunks are embedded using SentenceTransformers
    5. Embeddings are stored in ChromaDB

    **After upload**, you can ask questions about this document using /chat/ask

    **Supported formats:** PDF, TXT, DOCX (max 50MB)
    """
)
async def upload_document(
    file: UploadFile = File(..., description="Document to upload (PDF, TXT, or DOCX)"),
    current_user: dict = Depends(get_current_user)
):
    """
    Upload, process, and index a document for RAG.

    WHAT IS UploadFile?
        FastAPI's special class for handling multipart file uploads.
        The file is streamed to the server — we don't load it into
        memory until we call await file.read().

    WHAT IS File(...)?
        File() is FastAPI's way to declare a file parameter.
        The ... means it's REQUIRED (not optional).

    Args:
        file:         The uploaded file (multipart/form-data)
        current_user: Authenticated user (from JWT via Depends)

    Returns:
        DocumentUploadResponse with processing statistics
    """
    start_time = time.time()
    user_email = current_user["email"]

    logger.info(f"Document upload started | User: {user_email} | File: {file.filename}")

    
    # STEP 1: VALIDATE FILE TYPE
    
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have a filename."
        )

    if not is_allowed_file(file.filename, settings.allowed_extensions):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"File type not supported. "
                f"Allowed types: {', '.join(settings.allowed_extensions)}. "
                f"Your file: {file.filename}"
            )
        )

    
    # STEP 2: READ FILE CONTENT
    
    try:
        # Read ALL file bytes into memory
        # For large files, streaming would be better, but this is simpler for demo
        file_content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}"
        )

    
    # STEP 3: VALIDATE FILE SIZE
    
    file_size = len(file_content)

    if not document_processor.validate_file_size(file_size):
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File too large: {format_file_size(file_size)}. "
                f"Maximum allowed: {format_file_size(settings.max_file_size)}"
            )
        )

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    
    # STEP 4: PROCESS DOCUMENT (Extract text + chunk)
    
    document_id = generate_unique_id()
    safe_filename = create_unique_filename(file.filename)

    try:
        # document_processor:
        # 1. Detects file type (.pdf/.txt/.docx)
        # 2. Extracts text
        # 3. Splits into chunks
        # Returns: (list_of_chunks, metadata_dict)
        chunks, doc_metadata = await document_processor.process_document(
            file_content=file_content,
            filename=file.filename,
            document_id=document_id
        )

    except ValueError as e:
        # Known processing errors (empty file, wrong format, etc.)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Document processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )

    
    # STEP 5: STORE EMBEDDINGS IN CHROMADB
    
    try:
        # vector_store:
        # 1. Generates embeddings for all chunks (SentenceTransformers)
        # 2. Stores chunks + embeddings + metadata in ChromaDB
        # Returns: number of chunks stored
        chunks_stored = vector_store.store_document_chunks(
            chunks=chunks,
            document_metadata=doc_metadata,
            document_id=document_id
        )

    except Exception as e:
        logger.error(f"Vector storage failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store document embeddings: {str(e)}"
        )

    
    # STEP 6: SAVE TO DOCUMENT REGISTRY
    
    from datetime import datetime, timezone

    registry_entry = {
        "document_id": document_id,
        "filename": file.filename,
        "safe_filename": safe_filename,
        "file_type": doc_metadata["file_type"],
        "file_size": file_size,
        "file_size_human": format_file_size(file_size),
        "chunks_count": chunks_stored,
        "upload_date": datetime.now(timezone.utc).isoformat(),
        "uploaded_by": user_email,
        "status": "processed"
    }
    documents_registry[document_id] = registry_entry

    # Calculate total time
    elapsed = round(time.time() - start_time, 3)

    logger.info(
        f"Document processed successfully | "
        f"ID: {document_id} | "
        f"File: {file.filename} | "
        f"Chunks: {chunks_stored} | "
        f"Time: {elapsed}s"
    )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        status="processed",
        chunks_created=chunks_stored,
        file_size=format_file_size(file_size),
        message=f"Document processed and indexed successfully. {chunks_stored} chunks created."
    )


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List all uploaded documents",
    description="Returns a list of all documents that have been uploaded and indexed."
)
async def list_documents(
    current_user: dict = Depends(get_current_user),
    skip: int = Query(default=0, ge=0, description="Number of documents to skip (pagination)"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum documents to return")
):
    """
    List all uploaded documents with pagination.

    WHAT IS PAGINATION?
        If you have 1000 documents, you don't want to load them all at once.
        Pagination lets you get them in pages:
        - skip=0, limit=20  → first 20 documents
        - skip=20, limit=20 → next 20 documents
        - skip=40, limit=20 → next 20 documents

    Args:
        current_user: Authenticated user
        skip:         Number of documents to skip
        limit:        Max documents to return

    Returns:
        DocumentListResponse with paginated documents
    """
    all_docs = list(documents_registry.values())

    # Apply pagination
    paginated_docs = all_docs[skip: skip + limit]

    doc_items = [
        DocumentListItem(
            document_id=doc["document_id"],
            filename=doc["filename"],
            chunks_count=doc["chunks_count"],
            upload_date=doc["upload_date"],
            file_type=doc["file_type"]
        )
        for doc in paginated_docs
    ]

    logger.info(f"Documents listed for user: {current_user['email']} | Total: {len(all_docs)}")

    return DocumentListResponse(
        documents=doc_items,
        total=len(all_docs)
    )


@router.get(
    "/{document_id}",
    summary="Get document details",
    description="Get detailed information about a specific document."
)
async def get_document(
    document_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get details for a specific document.

    Args:
        document_id:  The document's unique ID
        current_user: Authenticated user

    Returns:
        Document metadata dict

    Raises:
        HTTPException 404: If document not found
    """
    doc = documents_registry.get(document_id)

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found. "
                   "It may have been deleted or never uploaded."
        )

    # Add current chunk count from ChromaDB
    try:
        chunks = vector_store.get_document_chunks(document_id)
        doc["current_chunk_count"] = len(chunks)
    except Exception:
        pass  # Non-critical

    return doc


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document",
    description="Delete a document and remove all its embeddings from the vector store."
)
async def delete_document(
    document_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a document and remove its embeddings from ChromaDB.

    WHY DELETE FROM CHROMADB TOO?
        If we only delete from the registry but NOT from ChromaDB,
        the AI would still find chunks from "deleted" documents
        when answering questions. That's confusing and wrong.

        We must delete from BOTH places.

    Args:
        document_id:  ID of document to delete
        current_user: Authenticated user

    Returns:
        DocumentDeleteResponse with deletion stats

    Raises:
        HTTPException 404: If document not found
    """
    # Check document exists
    if document_id not in documents_registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found."
        )

    doc = documents_registry[document_id]

    # Delete from ChromaDB (remove all chunk embeddings)
    chunks_deleted = vector_store.delete_document_chunks(document_id)

    # Remove from registry
    del documents_registry[document_id]

    logger.info(
        f"Document deleted | ID: {document_id} | "
        f"File: {doc['filename']} | "
        f"Chunks removed: {chunks_deleted}"
    )

    return DocumentDeleteResponse(
        document_id=document_id,
        message=f"Document '{doc['filename']}' deleted successfully.",
        chunks_deleted=chunks_deleted
    )


@router.get(
    "/stats/overview",
    summary="Get document storage statistics",
    description="Returns statistics about the document store and vector database."
)
async def get_stats(current_user: dict = Depends(get_current_user)):
    """
    Return statistics about the document store.

    Returns:
        Dict with counts and storage info
    """
    total_documents = len(documents_registry)
    total_chunks = sum(doc["chunks_count"] for doc in documents_registry.values())

    # Get ChromaDB stats
    chroma_stats = chroma_manager.get_collection_stats()

    return {
        "total_documents": total_documents,
        "total_chunks_in_registry": total_chunks,
        "vector_db_count": chroma_stats.get("count", 0),
        "supported_formats": settings.allowed_extensions,
        "max_file_size": format_file_size(settings.max_file_size),
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
