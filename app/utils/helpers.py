"""
==============================================================
app/utils/helpers.py — General Utility Functions
==============================================================

WHY THIS FILE EXISTS:
    A collection of small, reusable helper functions that don't
    belong to any specific layer (not authentication, not database,
    not AI). These are general-purpose utilities used across
    multiple files.

    Examples of what goes here:
    - File extension validation
    - Generating unique IDs
    - Sanitizing filenames
    - Formatting text
    - Measuring time

HOW IT CONNECTS:
    imported by: document_processor.py, documents_router.py,
                 rag_pipeline.py
==============================================================
"""

import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime

from app.utils.logger import logger


def generate_unique_id() -> str:
    """
    Generate a unique identifier (UUID4).

    WHY UUID?
        If two users upload files named "report.pdf", we need
        unique identifiers to tell them apart in the database.
        UUID4 generates random 128-bit IDs like:
        "a47f3d2c-8b9e-4f1a-b2c3-d4e5f6789abc"

        The probability of collision is astronomically low.

    Returns:
        A UUID4 string (36 characters including dashes)

    Example:
        doc_id = generate_unique_id()
        # doc_id = "a47f3d2c-8b9e-4f1a-b2c3-d4e5f6789abc"
    """
    return str(uuid.uuid4())


def get_file_extension(filename: str) -> str:
    """
    Extract the file extension from a filename.

    Args:
        filename: File name like "report.pdf" or "notes.txt"

    Returns:
        Lowercase extension including the dot: ".pdf", ".txt", ".docx"

    Examples:
        get_file_extension("report.pdf")   → ".pdf"
        get_file_extension("NOTES.TXT")    → ".txt"
        get_file_extension("noextension")  → ""
    """
    return Path(filename).suffix.lower()


def is_allowed_file(filename: str, allowed_extensions: list[str]) -> bool:
    """
    Check if a filename has an allowed extension.

    WHY?
        We don't want users uploading .exe files or .zip archives.
        We only accept .pdf, .txt, and .docx for our RAG pipeline.

    Args:
        filename:           Original filename from the upload
        allowed_extensions: List of allowed extensions [".pdf", ".txt", ".docx"]

    Returns:
        True if allowed, False if not

    Example:
        is_allowed_file("report.pdf", [".pdf", ".txt"])  → True
        is_allowed_file("virus.exe", [".pdf", ".txt"])   → False
    """
    ext = get_file_extension(filename)
    allowed = ext in allowed_extensions
    if not allowed:
        logger.warning(f"Rejected file with extension: {ext} | filename: {filename}")
    return allowed


def sanitize_filename(filename: str) -> str:
    """
    Clean a filename to prevent security issues.

    WHY?
        Filenames can contain dangerous characters:
        - "../../../etc/passwd" → Path traversal attack!
        - "file with spaces.pdf" → May cause issues
        - "file<script>.pdf" → XSS attempt

        We remove everything except letters, numbers, dots, dashes, underscores.

    Args:
        filename: Raw filename from user upload

    Returns:
        Safe, clean filename

    Example:
        sanitize_filename("../../../etc/passwd.txt") → "etcpasswd.txt"
        sanitize_filename("My Report (2024).pdf")    → "My_Report_2024.pdf"
    """
    # Get just the filename without directory path (prevents path traversal)
    filename = Path(filename).name

    # Replace spaces with underscores
    filename = filename.replace(" ", "_")

    # Remove any character that's not alphanumeric, dot, dash, or underscore
    # re.sub(pattern, replacement, string)
    filename = re.sub(r'[^\w\-_\.]', '', filename)

    # Prevent filenames from starting with a dot (hidden files)
    if filename.startswith('.'):
        filename = 'file' + filename

    # Ensure filename isn't empty
    if not filename or filename == '.':
        filename = f"upload_{generate_unique_id()[:8]}"

    logger.debug(f"Sanitized filename: {filename}")
    return filename


def create_unique_filename(original_filename: str) -> str:
    """
    Create a unique filename by prepending a timestamp and UUID.

    WHY?
        If two users upload "report.pdf", we need different names
        so the second doesn't overwrite the first.

    Args:
        original_filename: The uploaded file's original name

    Returns:
        Unique filename with timestamp prefix

    Example:
        create_unique_filename("report.pdf")
        → "20240115_143022_a47f3d2c_report.pdf"
    """
    safe_name = sanitize_filename(original_filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = generate_unique_id()[:8]
    return f"{timestamp}_{short_uuid}_{safe_name}"


def format_file_size(size_bytes: int) -> str:
    """
    Convert bytes to human-readable file size.

    Args:
        size_bytes: File size in bytes

    Returns:
        Human-readable string

    Examples:
        format_file_size(1024)        → "1.0 KB"
        format_file_size(1048576)     → "1.0 MB"
        format_file_size(1073741824)  → "1.0 GB"
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.1f} GB"


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate long text for display in logs or responses.

    Args:
        text:       Input text
        max_length: Maximum character length
        suffix:     What to append if truncated

    Returns:
        Truncated text with suffix if needed

    Example:
        truncate_text("A very long text here...", max_length=20)
        → "A very long text her..."
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def measure_time(start_time: float) -> float:
    """
    Calculate elapsed time in seconds since a start time.

    Used to measure how long API operations take.

    Args:
        start_time: Timestamp from time.time() when operation started

    Returns:
        Elapsed seconds as a float, rounded to 3 decimal places

    Example:
        start = time.time()
        # ... do some work ...
        elapsed = measure_time(start)
        # elapsed = 1.234  (seconds)
    """
    return round(time.time() - start_time, 3)


def clean_text(text: str) -> str:
    """
    Clean and normalize text for better embedding quality.

    WHY?
        Raw text from PDFs often has:
        - Extra whitespace between words
        - Newlines in the middle of sentences
        - Multiple consecutive spaces

        Cleaning this improves embedding quality and
        search accuracy.

    Args:
        text: Raw text to clean

    Returns:
        Cleaned, normalized text

    Example:
        clean_text("Hello   world\n\n  this   is   text")
        → "Hello world this is text"
    """
    if not text:
        return ""

    # Replace all whitespace sequences (spaces, tabs, newlines) with single space
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def chunk_text_by_sentences(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks for RAG.

    WHY CHUNKING?
        Language models have a context window limit — they can only
        process so much text at once. A 100-page PDF needs to be
        split into smaller pieces (chunks) that fit in the context.

    WHY OVERLAP?
        Imagine a sentence split across two chunks:
        Chunk 1: "The revenue increased significantly in Q3..."
        Chunk 2: "...due to new product launches in Asia."

        With overlap, both chunks contain part of the context,
        so neither chunk loses the full meaning.

    HOW OVERLAP WORKS:
        chunk_size=500, overlap=50 means:
        Chunk 1: characters 0-499
        Chunk 2: characters 450-949   ← starts 50 chars before chunk 1 ends
        Chunk 3: characters 900-1399  ← starts 50 chars before chunk 2 ends

    Args:
        text:       The full document text
        chunk_size: Maximum characters per chunk
        overlap:    Characters to repeat between adjacent chunks

    Returns:
        List of text chunks

    Example:
        chunks = chunk_text_by_sentences("Long document...", chunk_size=100, overlap=10)
        # chunks = ["First 100 chars...", "chars 90-189...", "chars 179-278...", ...]
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're near the end, take the rest
        if end >= text_length:
            chunk = text[start:text_length]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(clean_text(chunk))
            break

        # Try to split at a sentence boundary (period followed by space)
        # This creates more natural, readable chunks
        split_pos = text.rfind('. ', start, end)
        if split_pos != -1 and split_pos > start:
            # Found a sentence boundary — split there
            end = split_pos + 1  # Include the period
        else:
            # No sentence boundary found — try word boundary (space)
            split_pos = text.rfind(' ', start, end)
            if split_pos != -1 and split_pos > start:
                end = split_pos

        chunk = text[start:end]
        if chunk.strip():
            chunks.append(clean_text(chunk))

        # Move start forward, but overlap by `overlap` characters
        start = end - overlap

    logger.debug(f"Text split into {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks
