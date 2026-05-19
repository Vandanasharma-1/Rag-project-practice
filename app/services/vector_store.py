"""

app/services/vector_store.py — Vector Store Service


WHY THIS FILE EXISTS:
    This service handles the EMBEDDING and STORAGE of text chunks,
    and the RETRIEVAL of relevant chunks during question answering.

    It's the bridge between:
    - Raw text chunks (from document_processor.py)
    - ChromaDB storage (from chroma_manager.py)
    - The RAG pipeline (rag_pipeline.py uses this to find relevant chunks)

WHAT ARE EMBEDDINGS?
    An embedding converts text into a vector (list of numbers).
    This mathematical representation captures the SEMANTIC MEANING.

    "The dog ran fast" → [0.23, -0.11, 0.87, 0.45, 0.12, ...]
                          384 numbers for all-MiniLM-L6-v2 model

    SIMILAR texts produce SIMILAR vectors (mathematically close).
    DIFFERENT texts produce DIFFERENT vectors (mathematically far).

    This property enables "semantic search":
    Instead of keyword matching ("revenue" → find all docs with "revenue"),
    we find semantically SIMILAR meaning ("sales income" also matches "revenue").

WHAT IS SentenceTransformers?
    SentenceTransformers is a Python library that provides pre-trained
    models for converting text to embeddings.

    "all-MiniLM-L6-v2" model:
    - Small (22MB) and fast
    - Produces 384-dimensional vectors
    - Great for semantic similarity tasks
    - Runs locally, FREE, no API needed!

THE FULL EMBEDDING + STORAGE FLOW:
    Text chunks → SentenceTransformers → Vectors → ChromaDB storage

THE RETRIEVAL FLOW:
    User question → SentenceTransformers → Query vector
                                            ↓
                                    ChromaDB similarity search
                                            ↓
                                    Top-K most similar chunks

HOW IT CONNECTS:
    vector_store.py ← called by documents_router.py (store chunks)
    vector_store.py ← called by rag_pipeline.py (retrieve chunks)
    vector_store.py → uses chroma_manager.py (database operations)

"""

import time
from typing import List, Dict, Any, Optional, Tuple

from sentence_transformers import SentenceTransformer
import numpy as np

from app.database.chroma_manager import chroma_manager
from app.config.config import settings
from app.utils.logger import logger
from app.models.schemas import RetrievedChunk
from app.utils.helpers import generate_unique_id


class VectorStore:
    """
    Service for embedding text and performing semantic search.

    ARCHITECTURE PATTERN: Service Layer
        This class encapsulates all embedding and vector DB operations.
        The rest of the app doesn't need to know HOW embeddings work —
        it just calls these methods.

    LAZY LOADING:
        The embedding model is loaded on FIRST USE, not at startup.
        This speeds up application startup (SentenceTransformer download
        can take a few seconds).
    """

    def __init__(self):
        """
        Initialize VectorStore.

        NOTE: The embedding model is NOT loaded here (lazy loading).
        It loads on the first call to _get_embedding_model().
        This prevents slow startup if the model is cached locally.
        """
        self._embedding_model: Optional[SentenceTransformer] = None
        logger.info("VectorStore initialized (embedding model will load on first use)")

    def _get_embedding_model(self) -> SentenceTransformer:
        """
        Lazy-load the SentenceTransformers model.

        LAZY LOADING PATTERN:
            Check if model is already loaded. If yes, return it.
            If no, load it now and cache it for future calls.

            This is "lazy" because we delay loading until needed.

        FIRST RUN:
            The model downloads from HuggingFace Hub (~22MB).
            Subsequent runs use the cached local copy.

        Returns:
            The loaded SentenceTransformer model
        """
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            start_time = time.time()

            # SentenceTransformer downloads and caches the model
            # all-MiniLM-L6-v2 is a great balance of speed and quality
            self._embedding_model = SentenceTransformer(settings.embedding_model)

            load_time = round(time.time() - start_time, 2)
            logger.info(f"Embedding model loaded in {load_time}s")

        return self._embedding_model

    def generate_embedding(self, text: str) -> List[float]:
        """
        Convert a single text string to an embedding vector.

        HOW IT WORKS:
            1. Feed text through the neural network
            2. Get a 384-dimensional vector output
            3. Convert to Python list

        Args:
            text: The text to embed

        Returns:
            List of 384 floats (the embedding vector)

        Example:
            embedding = generate_embedding("Q3 revenue was $4.2B")
            # embedding = [0.23, -0.11, 0.87, ...] (384 numbers)
        """
        model = self._get_embedding_model()

        # encode() converts text to numpy array
        # tolist() converts numpy array to Python list (required for ChromaDB)
        embedding = model.encode(text).tolist()
        return embedding

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple texts to embeddings in one batch operation.

        WHY BATCH PROCESSING?
            Processing 50 chunks one-by-one takes much longer than
            processing all 50 at once. Batch processing uses GPU
            parallelism and is significantly faster.

            Single: 50 × 10ms = 500ms total
            Batch:  1 × 30ms = 30ms total  ← 16x faster!

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (one per text)
        """
        if not texts:
            return []

        model = self._get_embedding_model()

        logger.debug(f"Generating embeddings for {len(texts)} texts")
        start_time = time.time()

        # batch_size=32 processes 32 texts at a time
        # show_progress_bar=False prevents cluttering logs
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False
        ).tolist()

        elapsed = round(time.time() - start_time, 3)
        logger.info(f"Generated {len(embeddings)} embeddings in {elapsed}s")

        return embeddings

    def store_document_chunks(
        self,
        chunks: List[str],
        document_metadata: dict,
        document_id: str
    ) -> int:
        """
        Store document chunks with their embeddings in ChromaDB.

        WHAT HAPPENS HERE:
            For each text chunk:
            1. Generate embedding vector (384 numbers)
            2. Create a unique ID
            3. Attach metadata (filename, chunk index, etc.)
            4. Store all of this in ChromaDB

            ChromaDB stores:
            - id:        "doc_id_chunk_0", "doc_id_chunk_1", etc.
            - document:  "Revenue in Q3 was $4.2B..."  (raw text)
            - embedding: [0.23, -0.11, 0.87, ...]      (the vector)
            - metadata:  {"filename": "report.pdf", "chunk_index": 0}

        Args:
            chunks:            List of text chunks from document_processor
            document_metadata: Info about the source document
            document_id:       Unique ID of the document

        Returns:
            Number of chunks successfully stored

        Example:
            chunks = ["Revenue was $4.2B...", "Costs increased by 5%..."]
            metadata = {"filename": "report.pdf", "file_type": ".pdf"}
            count = store_document_chunks(chunks, metadata, "doc_abc123")
            # count = 2
        """
        if not chunks:
            logger.warning(f"No chunks to store for document {document_id}")
            return 0

        logger.info(f"Storing {len(chunks)} chunks for document: {document_id}")
        start_time = time.time()

        # Step 1: Generate all embeddings in one batch
        embeddings = self.generate_embeddings_batch(chunks)

        # Step 2: Prepare ChromaDB data structures
        # ChromaDB's add() expects parallel lists of equal length:
        # ids[0] corresponds to documents[0] corresponds to embeddings[0]
        ids = []
        documents = []
        metadatas = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create a unique ID for this chunk
            # Format: document_id + "_chunk_" + index
            chunk_id = f"{document_id}_chunk_{i}"

            ids.append(chunk_id)
            documents.append(chunk)

            # Metadata stored with each chunk
            # This is returned with search results for source attribution
            chunk_metadata = {
                **document_metadata,  # Include all document metadata
                "chunk_index": i,
                "chunk_id": chunk_id,
                # ChromaDB doesn't support lists in metadata, only primitives
                "total_chunks": int(document_metadata.get("total_chunks", len(chunks))),
            }
            metadatas.append(chunk_metadata)

        # Step 3: Store in ChromaDB
        # ChromaDB's upsert() is idempotent:
        # - If ID exists: UPDATE the existing entry
        # - If ID doesn't exist: INSERT a new entry
        # This means we can re-upload the same document safely
        collection = chroma_manager.collection
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        elapsed = round(time.time() - start_time, 3)
        logger.info(
            f"Stored {len(chunks)} chunks for {document_id} "
            f"in {elapsed}s | Total in DB: {collection.count()}"
        )

        return len(chunks)

    def search_similar_chunks(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[RetrievedChunk]:
        """
        Find the most relevant chunks for a user's question.

        THIS IS THE CORE OF RAG! This is what makes the system
        intelligent — finding the RIGHT information from your documents.

        HOW IT WORKS:
            1. Convert the user's question to an embedding vector
            2. ChromaDB searches for the CLOSEST vectors in the database
               (cosine similarity — mathematical distance between vectors)
            3. Returns the top_k closest chunks with their similarity scores

        WHAT IS COSINE SIMILARITY?
            Imagine each embedding as a direction in 384-dimensional space.
            Cosine similarity measures the ANGLE between two directions:
            - 0° angle (same direction) = similarity of 1.0 (identical meaning)
            - 90° angle = similarity of 0.0 (unrelated)
            - 180° angle = similarity of -1.0 (opposite meaning)

        Args:
            query:           The user's question
            top_k:           How many relevant chunks to return
            filter_metadata: Optional filter (e.g., only search specific docs)

        Returns:
            List of RetrievedChunk objects with text and similarity scores

        Example:
            chunks = search_similar_chunks("What was Q3 revenue?", top_k=5)
            # Returns 5 most relevant chunks from all stored documents
        """
        logger.info(f"Searching for top-{top_k} chunks matching: '{query[:100]}...'")
        start_time = time.time()

        # Step 1: Embed the query question
        query_embedding = self.generate_embedding(query)

        # Step 2: Query ChromaDB for similar vectors
        collection = chroma_manager.collection
        total_docs = collection.count()

        if total_docs == 0:
            logger.warning("No documents in vector store. Please upload documents first.")
            return []

        # Ensure top_k doesn't exceed available documents
        actual_top_k = min(top_k, total_docs)

        # ChromaDB query parameters:
        # query_embeddings: The search vector
        # n_results: How many results to return
        # include: What data to include in results
        # where: Optional metadata filter (e.g., {"document_id": "doc_123"})
        results = collection.query(
            query_embeddings=[query_embedding],  # List of queries (we have 1)
            n_results=actual_top_k,
            include=["documents", "metadatas", "distances"],
            where=filter_metadata if filter_metadata else None
        )

        # Step 3: Parse results into RetrievedChunk objects
        retrieved_chunks = []

        # results structure from ChromaDB:
        # {
        #   "ids": [["chunk_1", "chunk_2", ...]],        ← List of lists (one per query)
        #   "documents": [["text1", "text2", ...]],
        #   "metadatas": [[{...}, {...}, ...]],
        #   "distances": [[0.1, 0.3, ...]]               ← Lower = more similar
        # }
        # We have 1 query, so we take index [0] from each

        if not results["ids"] or not results["ids"][0]:
            logger.warning("No results found in vector store")
            return []

        for chunk_id, document, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            # Convert DISTANCE to SIMILARITY SCORE
            # ChromaDB with cosine metric returns distances: 0 = identical, 2 = opposite
            # We convert to similarity: 1 = identical, 0 = opposite
            # Formula: similarity = 1 - (distance / 2)
            similarity_score = round(1 - (distance / 2), 4)

            chunk = RetrievedChunk(
                chunk_id=chunk_id,
                content=document,
                relevance_score=similarity_score,
                document_id=str(metadata.get("document_id", "unknown")),
                filename=str(metadata.get("filename", "unknown")),
                chunk_index=int(metadata.get("chunk_index", 0))
            )
            retrieved_chunks.append(chunk)

        # Sort by relevance score (highest first)
        retrieved_chunks.sort(key=lambda x: x.relevance_score, reverse=True)

        elapsed = round(time.time() - start_time, 3)
        logger.info(
            f"Retrieved {len(retrieved_chunks)} chunks in {elapsed}s | "
            f"Top score: {retrieved_chunks[0].relevance_score if retrieved_chunks else 'N/A'}"
        )

        return retrieved_chunks

    def delete_document_chunks(self, document_id: str) -> int:
        """
        Delete all chunks belonging to a specific document.

        WHY?
            When a user deletes a document, we need to remove its
            chunks from ChromaDB too. Otherwise, the AI might still
            reference deleted documents.

        Args:
            document_id: The document whose chunks to delete

        Returns:
            Number of chunks deleted
        """
        logger.info(f"Deleting chunks for document: {document_id}")

        collection = chroma_manager.collection

        # Query to find all chunks for this document
        # We use the document_id field in metadata to identify them
        results = collection.get(
            where={"document_id": document_id},
            include=["documents"]
        )

        if not results["ids"]:
            logger.warning(f"No chunks found for document: {document_id}")
            return 0

        chunk_count = len(results["ids"])

        # Delete by their IDs
        collection.delete(ids=results["ids"])

        logger.info(f"Deleted {chunk_count} chunks for document: {document_id}")
        return chunk_count

    def get_document_chunks(self, document_id: str) -> List[Dict]:
        """
        Retrieve all stored chunks for a specific document.

        Useful for auditing what's in the vector store.

        Args:
            document_id: The document to retrieve chunks for

        Returns:
            List of chunk dicts with id, content, and metadata
        """
        collection = chroma_manager.collection
        results = collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"]
        )

        chunks = []
        for chunk_id, document, metadata in zip(
            results["ids"],
            results["documents"],
            results["metadatas"]
        ):
            chunks.append({
                "id": chunk_id,
                "content": document,
                "metadata": metadata
            })

        return chunks


# 
# SINGLETON INSTANCE
# 

vector_store = VectorStore()
