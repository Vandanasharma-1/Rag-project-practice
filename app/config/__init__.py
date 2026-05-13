"""
app/config/__init__.py

Makes `config` a Python package.
Exposes the settings object for convenient importing.

Usage:
    from app.config import settings
    # is equivalent to:
    from app.config.config import settings
"""
from app.config.config import settings, get_settings

__all__ = ["settings", "get_settings"]
