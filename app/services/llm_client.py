"""

app/services/llm_client.py — LLM Client (Google Gemini)


WHY THIS FILE EXISTS:
    This file manages communication with Google's Gemini AI model.

    The LLM (Large Language Model) is the "brain" that generates
    intelligent, natural language answers based on context.

WHAT IS A LLM?
    A Large Language Model is a neural network trained on massive
    amounts of text. It can:
    - Understand and generate human language
    - Answer questions when given context
    - Summarize, translate, explain, and reason

    We use Gemini because:
    - Has a generous FREE tier (perfect for learning)
    - Fast inference
    - High quality answers
    - Easy Python SDK

WHAT IS PROMPT ENGINEERING?
    The "prompt" is the text we send to the LLM. Good prompts produce
    better answers. We structure our prompt as:

    System prompt (instructions for the AI):
    "You are an enterprise AI assistant. Answer using ONLY the context provided..."

    User prompt (the actual question with context):
    "Context: [retrieved document chunks]
     Question: What was revenue in Q3?"

    This structure guides the AI to:
    - Only use information from our documents (not its training data)
    - Acknowledge when it doesn't know something
    - Format answers professionally

HOW IT CONNECTS:
    llm_client.py ← called by rag_pipeline.py
    llm_client.py → calls Google Gemini API

"""

import time
from typing import Optional

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from app.config.config import settings
from app.utils.logger import logger


class LLMClient:
    """
    Client for interacting with Google Gemini LLM.

    DESIGN PATTERN: Adapter Pattern
        This class "adapts" the raw Gemini API to our application's
        interface. If we ever switch to OpenAI or another LLM, we
        only change THIS file — the rest of the app is unaffected.

    LAZY INITIALIZATION:
        The Gemini model is configured once and reused for all requests.
        We initialize it on first use to avoid startup errors if the
        API key is not yet configured.
    """

    def __init__(self):
        """Initialize LLM client (model loads on first use)."""
        self._model = None
        self._initialized = False
        logger.info("LLMClient initialized (Gemini model will load on first use)")

    def _initialize_model(self) -> None:
        """
        Configure and initialize the Gemini model.

        WHAT IS SAFETY SETTINGS?
            Gemini has built-in content filters that can block responses.
            For an enterprise document QA system, we set them to
            BLOCK_NONE to allow all business-related content to pass through.

        WHAT IS GENERATION CONFIG?
            Parameters that control how the model generates text:
            - temperature: 0.0 = deterministic, 1.0 = creative/random
                          We use 0.3 — factual but with some flexibility
            - max_output_tokens: Maximum length of the response
            - top_p: Nucleus sampling parameter (diversity control)
        """
        try:
            logger.info(f"Initializing Gemini model: {settings.gemini_model}")

            # Configure the API key for all subsequent Gemini calls
            genai.configure(api_key=settings.gemini_api_key)

            # Safety settings — for enterprise document QA, we want minimal filtering
            # BLOCK_NONE = Don't block based on this category
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            # Generation configuration
            generation_config = genai.GenerationConfig(
                temperature=0.3,        # Low = more factual, less creative
                max_output_tokens=2048, # Max response length (~1500 words)
                top_p=0.8,              # Consider top 80% most likely tokens
                top_k=40,               # Consider top 40 candidate tokens
            )

            # Initialize the model
            self._model = genai.GenerativeModel(
                model_name=settings.gemini_model,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )

            self._initialized = True
            logger.info(f"Gemini model '{settings.gemini_model}' initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise

    def _get_model(self):
        """
        Lazy-load the Gemini model.

        Returns the model, initializing it on first call.
        """
        if not self._initialized or self._model is None:
            self._initialize_model()
        return self._model

    def build_rag_prompt(self, question: str, context_chunks: list) -> str:
        """
        Build a well-structured prompt for RAG question answering.

        WHAT IS PROMPT ENGINEERING?
            The prompt is the instruction we give to the LLM.
            A good prompt:
            1. Gives the model a clear ROLE ("You are an enterprise assistant")
            2. Provides the CONTEXT (relevant document chunks)
            3. States the TASK (answer this question)
            4. Sets CONSTRAINTS (only use provided context)
            5. Specifies OUTPUT FORMAT (professional, concise)

        WHY "ONLY USE PROVIDED CONTEXT"?
            This is crucial for enterprise use! Without this constraint,
            the model might use its training data to answer, which could:
            - Be outdated
            - Contain incorrect information
            - Hallucinate facts

            By constraining to provided context, answers are GROUNDED
            in your actual documents.

        Args:
            question:       The user's question
            context_chunks: List of relevant text chunks (from vector search)

        Returns:
            Formatted prompt string ready to send to Gemini
        """
        # Format context chunks for the prompt
        # Number them for clarity
        if not context_chunks:
            context_str = "No relevant context found in the uploaded documents."
        else:
            context_parts = []
            for i, chunk in enumerate(context_chunks, 1):
                # Each chunk includes its source for transparency
                context_parts.append(
                    f"[Context {i} - Source: {chunk.filename}, "
                    f"Relevance: {chunk.relevance_score:.2f}]\n"
                    f"{chunk.content}"
                )
            context_str = "\n\n---\n\n".join(context_parts)

        # The full prompt with system instructions + context + question
        prompt = f"""You are an expert Enterprise AI Assistant for document analysis and question answering.

Your task is to answer questions based STRICTLY on the provided context from uploaded enterprise documents.

IMPORTANT RULES:
1. Answer ONLY based on the provided context below
2. If the context doesn't contain enough information, say: "The uploaded documents don't contain sufficient information to answer this question."
3. Always cite which context source(s) you used in your answer
4. Be precise, professional, and concise
5. Do NOT use external knowledge or make up information
6. If numbers or dates are mentioned, quote them exactly as they appear

=== CONTEXT FROM DOCUMENTS ===
{context_str}

=== QUESTION ===
{question}

=== ANSWER ===
Please provide a clear, well-structured answer based on the context above:"""

        return prompt

    def generate_answer(self, question: str, context_chunks: list) -> str:
        """
        Generate an answer to the user's question using Gemini + context.

        FULL RAG ANSWER GENERATION FLOW:
            1. Build a prompt with the question + retrieved context
            2. Send prompt to Gemini API
            3. Gemini reads context, understands question, generates answer
            4. Return the answer text

        This function is the LAST step in the RAG pipeline.
        It synthesizes all the retrieved chunks into a coherent answer.

        Args:
            question:       The user's question
            context_chunks: Retrieved RetrievedChunk objects

        Returns:
            Generated answer string from Gemini

        Raises:
            Exception: If API call fails or response is blocked
        """
        model = self._get_model()

        logger.info(f"Generating answer for question: '{question[:100]}...'")
        start_time = time.time()

        # Build the structured prompt
        prompt = self.build_rag_prompt(question, context_chunks)

        try:
            # Send the prompt to Gemini
            # generate_content() is a synchronous API call
            # It returns when the full response is ready
            response = model.generate_content(prompt)

            # Check if the response was blocked by safety filters
            if not response.candidates:
                logger.warning("Gemini response was blocked by safety filters")
                return "I'm unable to answer this question due to content policy restrictions."

            # Check finish reason
            finish_reason = response.candidates[0].finish_reason
            if finish_reason.name == "SAFETY":
                logger.warning("Response blocked due to safety settings")
                return "This question could not be answered due to safety policy restrictions."

            # Extract the text from the response
            answer = response.text

            elapsed = round(time.time() - start_time, 3)
            logger.info(
                f"Answer generated in {elapsed}s | "
                f"Input tokens: ~{len(prompt)//4} | "
                f"Output length: {len(answer)} chars"
            )

            return answer

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def generate_simple_response(self, prompt: str) -> str:
        """
        Generate a response to any prompt (not RAG-specific).

        Used for health checks, testing, and simple generation tasks.

        Args:
            prompt: Any text prompt

        Returns:
            Generated response string
        """
        model = self._get_model()

        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Simple generation failed: {e}")
            raise

    def health_check(self) -> bool:
        """
        Check if Gemini API is accessible.

        Sends a minimal test prompt to verify:
        1. API key is valid
        2. Network connection to Google is working
        3. Model is responsive

        Returns:
            True if healthy, False otherwise
        """
        try:
            model = self._get_model()
            # A minimal test prompt to verify connectivity
            response = model.generate_content("Say 'OK' if you can hear me.")
            return bool(response.text)
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return False


# 
# SINGLETON INSTANCE
# 

llm_client = LLMClient()
