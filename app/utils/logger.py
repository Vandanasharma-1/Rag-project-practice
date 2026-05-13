"""
==============================================================
app/utils/logger.py — Centralized Logging Setup
==============================================================

WHY THIS FILE EXISTS:
    Logging is how your application "talks to you" while running.
    Instead of print() statements scattered everywhere, we use
    a proper logging system that:
    - Timestamps every message
    - Colors messages by severity (INFO=green, ERROR=red)
    - Writes logs to BOTH console AND disk files
    - Rotates log files so they don't fill up your hard drive

WHAT IS LOGURU?
    Loguru is a beautiful, zero-config logging library for Python.
    It's much simpler than Python's built-in logging module.

    Usage example:
        from app.utils.logger import logger
        logger.info("User logged in: john@example.com")
        logger.error("Database connection failed!")
        logger.debug("Processing chunk 3 of 20...")

LOG LEVELS (from least to most severe):
    DEBUG   → Very detailed info (only in development)
    INFO    → Normal operations ("User uploaded file")
    WARNING → Something unexpected but not broken ("Slow query")
    ERROR   → Something failed ("File not found")
    CRITICAL → App might crash ("Out of memory")

HOW IT CONNECTS:
    Every other file imports logger from here:
    from app.utils.logger import logger
==============================================================
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logging(log_level: str = "INFO", log_dir: str = "./data/logs", log_file: str = "app.log") -> None:
    """
    Configure Loguru logging for the entire application.

    This function sets up:
    1. Console output (colored, human-readable)
    2. File output (for persistent logs you can review later)
    3. Log rotation (files don't grow forever)

    Args:
        log_level: Minimum severity level to log (DEBUG/INFO/WARNING/ERROR)
        log_dir:   Directory where log files will be stored
        log_file:  Name of the log file

    BEGINNER TIP:
        "Rotation" means: when a log file reaches 10MB, create a new one.
        "Retention" means: delete log files older than 7 days.
        This prevents logs from filling your entire hard drive!
    """

    # Step 1: Remove the default Loguru handler
    # Loguru starts with a basic console handler — we replace it with our custom ones
    logger.remove()

    # Step 2: Add a CONSOLE handler
    # This writes colorful logs to your terminal while the app runs
    # Format breakdown:
    #   {time}     → Timestamp like "2024-01-15 14:30:22"
    #   {level}    → Log level like "INFO" or "ERROR"
    #   {name}     → Module name like "app.services.rag_pipeline"
    #   {function} → Function name where logger was called
    #   {line}     → Line number in the source file
    #   {message}  → The actual log message
    logger.add(
        sys.stdout,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,      # Enable colors in terminal
        backtrace=True,     # Show full traceback for errors
        diagnose=True,      # Show variable values in tracebacks (DISABLE in production)
    )

    # Step 3: Ensure log directory exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Step 4: Add a FILE handler
    # This writes logs to a file on disk for later review
    log_file_path = log_path / log_file
    logger.add(
        str(log_file_path),
        level=log_level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        rotation="10 MB",      # Create new file when this one reaches 10MB
        retention="7 days",    # Delete files older than 7 days
        compression="zip",     # Compress old log files to save space
        backtrace=True,
        diagnose=False,        # Don't expose variable values in file logs (security)
        enqueue=True,          # Thread-safe logging (important for async apps!)
    )

    logger.info(f"Logging initialized | Level: {log_level} | File: {log_file_path}")


# The `logger` object is already available from loguru
# After calling setup_logging(), it will use our custom configuration
# Other files just import this logger directly:
#   from app.utils.logger import logger

__all__ = ["logger", "setup_logging"]
