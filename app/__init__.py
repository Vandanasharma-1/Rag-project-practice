"""
app/__init__.py — Application Package Initializer

WHY THIS FILE EXISTS:
    Python requires an __init__.py file in every directory
    that should be treated as a "package" (importable module).

    Without this file, Python would NOT recognize the `app`
    directory as a package, and imports like:
        from app.config.config import settings
    would FAIL with ModuleNotFoundError.
"""

__version__ = "1.0.0"
__author__ = "Enterprise RAG Team"
