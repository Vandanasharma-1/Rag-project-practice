"""
==============================================================
app/routers/chat_router.py — Chat/RAG API Routes
==============================================================

WHY THIS FILE EXISTS:
    This is where users interact with the AI assistant.
    It exposes the RAG pipeline via HTTP endpoints.

    Endpoints:
    - POST /chat/ask         → Ask a question (main RAG endpoint)
    - POST /chat/search      → Search for relevant chunks (no LLM)
    - GET  /chat/history     → Get conversation history (demo only)

THE MAIN ENDPOINT: POST /chat/ask
    This is the core of the entire application.
    User sends a question → Gets an AI answer backed by documents.

    FULL REQUEST LIFECYCLE:
    1. HTTP POST /chat/ask with body: {"question": "What was Q3 revenue?"}
    2. JWT middleware validates the Authorization token
    3. chat_router receives the validated request
    4. Passes question to rag_pipeline.ask()
    5. rag_pipeline:
       a. Embeds the question → [0.23, -0.11, ...]
       b. Searches ChromaDB for similar chunks
       c. Builds prompt with question + chunks
       d. Calls Gemini API
       e. Returns ChatResponse
    6. router returns the ChatResponse as JSON

HOW IT CONNECTS:
    chat_router.py ← called by HTTP clients
    chat_router.py → uses rag_pipeline.py (the full RAG flow)
    chat_router.py → uses auth_router.py (get_current_user)
==============================================================
"""

import time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query

from app.models.schemas import ChatRequest, ChatResponse, RetrievedChunk, ErrorResponse
from app.services.rag_pipeline import rag_pipeline
from app.routers.auth_router import get_current_user
from app.config.config import settings
from app.utils.logger import logger

# ==============================================================
# ROUTER SETUP
# ==============================================================

router = APIRouter(
    prefix="/chat",
    tags=["Chat & RAG"],
    responses={
        401: {"model": ErrorResponse, "description": "JWT token required"},
        503: {"description": "AI service temporarily unavailable"},
    }
)

# ==============================================================
# IN-MEMORY CONVERSATION HISTORY
# ==============================================================
# ⚠️ DEMO ONLY: In production, use a database!
# Stores conversation history per user: {email: [messages]}
conversation_history: dict = {}


# ==============================================================
# API ROUTES
# ==============================================================

@router.post(
    "/ask",
    response_model=ChatResponse,
    summary="Ask a question to the RAG assistant",
    description="""
    Ask a natural language question. The system will:
    1. Search through uploaded documents for relevant context
    2. Retrieve the top-K most relevant document chunks
    3. Generate a contextual answer using Google Gemini AI
    4. Return the answer with source citations and relevance scores

    **Prerequisites:** Upload at least one document via `/documents/upload` first.

    **Example question types:**
    - "What were the key financial metrics in Q3 2023?"
    - "Summarize the main risks mentioned in the report"
    - "What does the policy say about remote work?"
    - "Who are the key stakeholders mentioned?"
    """,
)
async def ask_question(
    chat_request: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    The main RAG endpoint — ask a question, get an AI-generated answer.

    WHAT HAPPENS HERE:
        1. Receive and validate the question
        2. Pass to RAG pipeline (which handles everything)
        3. Store in conversation history
        4. Return the full ChatResponse

    Args:
        chat_request: Contains the question and optional top_k
        current_user: Authenticated user (from JWT)

    Returns:
        ChatResponse with answer, retrieved chunks, and metadata

    ChatResponse fields:
        - question:                The original question
        - answer:                  AI-generated answer
        - retrieved_chunks:        Source document chunks used
        - processing_time_seconds: How long it took
        - model_used:              "gemini-1.5-flash"
        - chunks_retrieved:        Number of chunks found
        - status:                  "success", "no_context", or "error"
    """
    user_email = current_user["email"]
    question = chat_request.question
    top_k = chat_request.top_k or settings.top_k

    logger.info(
        f"Chat request | User: {user_email} | "
        f"Question: '{question[:80]}...' | "
        f"Top-K: {top_k}"
    )

    try:
        # Run the complete RAG pipeline
        # This is an async call — the server can handle other requests
        # while waiting for embeddings and Gemini API
        response = await rag_pipeline.ask(
            question=question,
            top_k=top_k
        )

        # Store in conversation history (for /history endpoint)
        _store_conversation(user_email, question, response.answer)

        logger.info(
            f"Chat response sent | User: {user_email} | "
            f"Status: {response.status} | "
            f"Time: {response.processing_time_seconds}s"
        )

        return response

    except Exception as e:
        logger.error(f"Chat pipeline error for user {user_email}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"The AI service encountered an error: {str(e)}. Please try again."
        )


@router.post(
    "/search",
    response_model=List[RetrievedChunk],
    summary="Search documents without AI generation",
    description="""
    Semantic search through uploaded documents without calling the LLM.
    Returns the most relevant document chunks for a query.

    Useful for:
    - Debugging: See what documents would be retrieved
    - Browsing: Explore document content
    - Testing: Evaluate retrieval quality
    """
)
async def search_documents(
    query: str = Query(..., min_length=3, description="Search query"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of results"),
    current_user: dict = Depends(get_current_user)
):
    """
    Semantic search without LLM generation.

    This is cheaper and faster than /ask because it only does
    vector search — no Gemini API call.

    Args:
        query:        Search query string
        top_k:        Number of results to return
        current_user: Authenticated user

    Returns:
        List of RetrievedChunk objects with relevance scores
    """
    logger.info(f"Document search | User: {current_user['email']} | Query: '{query[:80]}'")

    try:
        chunks = await rag_pipeline.get_relevant_chunks_only(
            question=query,
            top_k=top_k
        )
        return chunks
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get(
    "/history",
    summary="Get conversation history",
    description="Returns the last 20 messages in your conversation history."
)
async def get_conversation_history(
    current_user: dict = Depends(get_current_user),
    limit: int = Query(default=20, ge=1, le=100)
):
    """
    Return conversation history for the current user.

    Args:
        current_user: Authenticated user
        limit:        Max messages to return

    Returns:
        List of conversation messages with timestamps
    """
    user_email = current_user["email"]
    history = conversation_history.get(user_email, [])

    # Return most recent messages first
    recent = history[-limit:][::-1]

    return {
        "user": user_email,
        "total_messages": len(history),
        "messages": recent
    }


@router.delete(
    "/history",
    summary="Clear conversation history",
    description="Delete all conversation history for the current user."
)
async def clear_conversation_history(
    current_user: dict = Depends(get_current_user)
):
    """
    Clear the conversation history for the current user.

    Args:
        current_user: Authenticated user

    Returns:
        Confirmation message
    """
    user_email = current_user["email"]
    count = len(conversation_history.get(user_email, []))
    conversation_history[user_email] = []

    logger.info(f"Cleared {count} messages for user: {user_email}")

    return {
        "message": f"Cleared {count} conversation messages.",
        "user": user_email
    }


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def _store_conversation(user_email: str, question: str, answer: str) -> None:
    """
    Store a question-answer pair in conversation history.

    Args:
        user_email: The user's email (conversation key)
        question:   The user's question
        answer:     The AI's answer
    """
    from datetime import datetime, timezone

    if user_email not in conversation_history:
        conversation_history[user_email] = []

    conversation_history[user_email].append({
        "role": "user",
        "content": question,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    conversation_history[user_email].append({
        "role": "assistant",
        "content": answer[:500] + ("..." if len(answer) > 500 else ""),  # Truncate for storage
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    # Keep only last 200 messages
    if len(conversation_history[user_email]) > 200:
        conversation_history[user_email] = conversation_history[user_email][-200:]
