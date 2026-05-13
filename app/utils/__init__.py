"""
app/utils/__init__.py

Makes `utils` a Python package.
Exposes utility functions and the logger.

WHAT ARE UTILITIES?
    Utility functions are small, reusable helpers that don't
    belong to any specific business domain. Examples:
    - Generating unique IDs
    - Formatting file sizes
    - Hashing passwords
    - Logging setup

Usage:
    from app.utils import logger
    from app.utils.helpers import generate_unique_id
    from app.utils.auth import hash_password, create_access_token
"""
from app.utils.logger import logger, setup_logging

__all__ = ["logger", "setup_logging"]
