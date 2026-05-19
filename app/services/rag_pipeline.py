"""

app/services/rag_pipeline.py — RAG Pipeline Orchestrator


WHY THIS FILE EXISTS:
    This is the BRAIN of the entire system — it orchestrates the
    complete Retrieval-Augmented Generation (RAG) pipeline.

    When a user asks a question, THIS file coordinates:
    1. Vector search (find relevant chunks)
    2. Context building (format chunks for the LLM)
    3. LLM generation (get Gemini to answer)
    4. Response assembly (combine everything)

WHAT IS RAG?
    RAG = Retrieval-Augmented Generation

    Traditional LLM approach (WITHOUT RAG):
        User: "What was Apple's revenue in Q3 2023?"
        LLM: Makes up an answer or says "I don't know"
        Problem: LLM doesn't have YOUR documents!

    RAG approach (WITH our system):
        User: "What was Apple's revenue in Q3 2023?"
             ↓
        1. RETRIEVE: Find relevant chunks from your Apple Q3 report
        2. AUGMENT:  Add those chunks to the LLM's context
        3. GENERATE: LLM answers using the retrieved context

        LLM: "Based on the uploaded Q3 2023 report, Apple's revenue
              was $89.5 billion, representing a 1% year-over-year decline."
        ✓ Accurate, ✓ Source-backed, ✓ From YOUR documents!

THE COMPLETE RAG FLOW:
    
                        QUESTION FLOW                        
                                                             
      User Question                                          
                                                            
           ▼                                                 
      [1] Embed Question (SentenceTransformers)              
            "What was Q3 revenue?"                         
            → [0.23, -0.11, 0.87, ...]                     
                                                            
           ▼                                                 
      [2] Vector Search (ChromaDB)                          
            Find top-K most similar chunks                 
            → 5 chunks with relevance scores               
                                                            
           ▼                                                 
      [3] Build Prompt (Prompt Engineering)                  
            "Context: [chunk1] [chunk2] ...                
             Question: What was Q3 revenue?"               
                                                            
           ▼                                                 
      [4] LLM Generation (Gemini)                           
            Answer based on context                        
                                                            
           ▼                                                 
      [5] Return Response                                    
           answer + chunks + scores + metadata              
    

HOW IT CONNECTS:
    rag_pipeline.py ← called by chat_router.py
    rag_pipeline.py → uses vector_store.py (retrieve chunks)
    rag_pipeline.py → uses llm_client.py (generate answer)

"""

import time
from typing import List, Optional

from app.services.vector_store import vector_store
from app.services.llm_client import llm_client
from app.models.schemas import ChatResponse, RetrievedChunk
from app.config.config import settings
from app.utils.logger import logger


class RAGPipeline:
    """
    Orchestrates the complete RAG (Retrieval-Augmented Generation) pipeline.

    This is a stateless orchestrator — it coordinates vector_store
    and llm_client but doesn't maintain its own state.

    DESIGN PATTERN: Facade Pattern
        Complex subsystems (embedding, vector search, LLM generation)
        are hidden behind a simple interface: ask(question).
        The router just calls ask() and gets a complete response.
    """

    def __init__(self):
        """Initialize the RAG pipeline with its dependencies."""
        self.vector_store = vector_store
        self.llm_client = llm_client
        logger.info("RAGPipeline initialized")

    async def ask(
        self,
        question: str,
        top_k: int = None,
        document_id_filter: Optional[str] = None
    ) -> ChatResponse:
        """
        The main RAG pipeline: answer a question using document context.

        STEP-BY-STEP WALKTHROUGH:
            1. Record start time (for performance measurement)
            2. Retrieve relevant chunks from vector DB
            3. Check if any chunks were found
            4. Generate answer using LLM + chunks
            5. Assemble and return complete response

        Args:
            question:           The user's natural language question
            top_k:              How many chunks to retrieve (default: settings.top_k)
            document_id_filter: Optional: restrict search to specific document

        Returns:
            ChatResponse with answer, chunks, scores, and metadata

        Example:
            response = await pipeline.ask("What was Q3 revenue?", top_k=5)
            print(response.answer)          # "Q3 revenue was $4.2B..."
            print(response.chunks_retrieved) # 5
            print(response.processing_time_seconds) # 2.3
        """
        # Start timing the full pipeline
        pipeline_start = time.time()

        logger.info(f"RAG Pipeline starting for question: '{question[:100]}'")

        # Use default top_k from settings if not specified
        k = top_k or settings.top_k

        # 
        # STEP 1: RETRIEVE RELEVANT CHUNKS
        # 
        logger.info(f"Step 1: Retrieving top-{k} relevant chunks")

        # Optional: Filter to a specific document
        metadata_filter = None
        if document_id_filter:
            metadata_filter = {"document_id": document_id_filter}
            logger.info(f"Filtering to document: {document_id_filter}")

        # Vector similarity search
        retrieved_chunks = self.vector_store.search_similar_chunks(
            query=question,
            top_k=k,
            filter_metadata=metadata_filter
        )

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")

        # 
        # STEP 2: HANDLE EMPTY RESULTS
        # 
        if not retrieved_chunks:
            # No documents uploaded yet
            elapsed = time.time() - pipeline_start

            logger.warning("No relevant chunks found. No documents may be uploaded.")

            return ChatResponse(
                question=question,
                answer=(
                    "I couldn't find any relevant information to answer your question. "
                    "This could mean:\n"
                    "1. No documents have been uploaded yet. Please upload relevant documents first.\n"
                    "2. The uploaded documents don't contain information related to your question.\n"
                    "3. Try rephrasing your question.\n\n"
                    "Use the /api/v1/documents/upload endpoint to upload PDF, TXT, or DOCX files."
                ),
                retrieved_chunks=[],
                processing_time_seconds=round(elapsed, 3),
                model_used=settings.gemini_model,
                chunks_retrieved=0,
                status="no_context"
            )

        # 
        # STEP 3: FILTER LOW-RELEVANCE CHUNKS
        # 
        # Remove chunks with very low relevance scores
        # A score below 0.3 is likely not related to the question
        MIN_RELEVANCE_SCORE = 0.3
        relevant_chunks = [
            chunk for chunk in retrieved_chunks
            if chunk.relevance_score >= MIN_RELEVANCE_SCORE
        ]

        if not relevant_chunks:
            logger.warning(
                f"All {len(retrieved_chunks)} chunks scored below {MIN_RELEVANCE_SCORE}. "
                "Question may not match document content."
            )
            relevant_chunks = retrieved_chunks  # Use them anyway, LLM will handle it

        logger.info(
            f"Using {len(relevant_chunks)} chunks above relevance threshold "
            f"(scores: {[round(c.relevance_score, 3) for c in relevant_chunks]})"
        )

        # 
        # STEP 4: GENERATE ANSWER WITH GEMINI
        # 
        logger.info("Step 2: Generating answer with Gemini")
        llm_start = time.time()

        try:
            answer = self.llm_client.generate_answer(
                question=question,
                context_chunks=relevant_chunks
            )
            llm_elapsed = round(time.time() - llm_start, 3)
            logger.info(f"LLM generation completed in {llm_elapsed}s")

        except Exception as e:
            # If Gemini fails, return a graceful error
            logger.error(f"LLM generation failed: {e}")
            elapsed = round(time.time() - pipeline_start, 3)

            return ChatResponse(
                question=question,
                answer=f"I encountered an error while generating the answer: {str(e)}",
                retrieved_chunks=retrieved_chunks,
                processing_time_seconds=elapsed,
                model_used=settings.gemini_model,
                chunks_retrieved=len(retrieved_chunks),
                status="error",
                error_message=str(e)
            )

        # 
        # STEP 5: ASSEMBLE COMPLETE RESPONSE
        # 
        total_elapsed = round(time.time() - pipeline_start, 3)

        response = ChatResponse(
            question=question,
            answer=answer,
            retrieved_chunks=relevant_chunks,
            processing_time_seconds=total_elapsed,
            model_used=settings.gemini_model,
            chunks_retrieved=len(relevant_chunks),
            status="success"
        )

        logger.info(
            f"RAG Pipeline complete | "
            f"Total time: {total_elapsed}s | "
            f"Chunks used: {len(relevant_chunks)} | "
            f"Answer length: {len(answer)} chars"
        )

        return response

    async def get_relevant_chunks_only(
        self,
        question: str,
        top_k: int = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks WITHOUT generating an answer.

        Useful for:
        - Debugging: See what the system would retrieve
        - Transparency: Show users what sources are available
        - Testing: Evaluate retrieval quality before LLM costs

        Args:
            question: The search query
            top_k:    Number of chunks to retrieve

        Returns:
            List of retrieved chunks with relevance scores
        """
        k = top_k or settings.top_k
        return self.vector_store.search_similar_chunks(
            query=question,
            top_k=k
        )


# 
# SINGLETON INSTANCE
# 

rag_pipeline = RAGPipeline()
