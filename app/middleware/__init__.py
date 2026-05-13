"""
app/middleware/__init__.py

Makes `middleware` a Python package.
Exposes middleware classes for use in main.py.

WHAT IS MIDDLEWARE?
    Middleware is code that runs for EVERY request and response,
    automatically, without being called explicitly in each route.

    Think of it as a pipeline that every request must pass through:
    Request → Middleware1 → Middleware2 → Route Handler → Response

Usage:
    from app.middleware.request_logger import RequestLoggingMiddleware
    app.add_middleware(RequestLoggingMiddleware)
"""
from app.middleware.request_logger import RequestLoggingMiddleware

__all__ = ["RequestLoggingMiddleware"]
