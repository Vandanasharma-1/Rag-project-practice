"""
==============================================================
app/middleware/request_logger.py — HTTP Request Logging Middleware
==============================================================

WHY THIS FILE EXISTS:
    Middleware is code that runs for EVERY REQUEST and RESPONSE,
    automatically, without you having to call it manually.

    This middleware:
    1. Logs every incoming HTTP request
    2. Measures how long each request takes
    3. Logs the HTTP status code of each response
    4. Assigns a unique Request ID to track requests

WHAT IS MIDDLEWARE?
    Think of middleware as a "wrapper" around your API routes.

    Without middleware:
        Request → Router → Response

    With middleware:
        Request → [Middleware START] → Router → [Middleware END] → Response

    Our middleware:
        1. Catches the request BEFORE the router handles it
        2. Logs request details (method, URL, IP address)
        3. Passes the request to the router
        4. Catches the response AFTER the router handles it
        5. Logs response details (status code, time taken)

WHY LOG REQUESTS?
    In production, you need to know:
    - How many requests per second?
    - Which endpoints are slow?
    - What HTTP errors are occurring?
    - Which IP addresses are making requests?

    This is essential for debugging and monitoring!

UNIQUE REQUEST ID:
    Each request gets a unique UUID. This appears in every log
    line for that request. When debugging, you can search logs
    for a specific request ID and see everything that happened.

HOW IT CONNECTS:
    request_logger.py → added to FastAPI app in main.py
    Runs automatically for EVERY route without any changes needed
==============================================================
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.utils.logger import logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs all HTTP requests and responses.

    WHAT IS ASGI?
        ASGI = Asynchronous Server Gateway Interface
        It's the protocol that connects our Python app to the web server.
        FastAPI is built on ASGI (via Starlette).

    WHAT IS BaseHTTPMiddleware?
        A base class from Starlette that makes it easy to write
        middleware. We just override dispatch() to add our logic.

    HOW dispatch() WORKS:
        def dispatch(request, call_next):
            # Code here runs BEFORE the route handler
            response = await call_next(request)  # ← This calls the actual route
            # Code here runs AFTER the route handler
            return response
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process each request/response cycle with logging.

        Args:
            request:   The incoming HTTP request
            call_next: Function to call the next handler (the actual route)

        Returns:
            The HTTP response
        """
        # Generate a unique request ID for tracking
        # This appears in all log messages for this request
        request_id = str(uuid.uuid4())[:8]  # Short 8-char ID for readability

        # Record start time for performance measurement
        start_time = time.time()

        # Extract useful request information
        method = request.method           # GET, POST, PUT, DELETE
        url = str(request.url)            # Full URL including query params
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")

        # Log the INCOMING request
        logger.info(
            f"[{request_id}] ← {method} {url} | "
            f"IP: {client_ip} | "
            f"Agent: {user_agent[:50]}"  # Truncate long user agents
        )

        # Store request_id in request state so route handlers can access it
        # Example: request.state.request_id in a route function
        request.state.request_id = request_id

        try:
            # Call the actual route handler
            # This is where your @app.get("/endpoint") code runs
            response = await call_next(request)

            # Calculate total processing time
            elapsed_ms = round((time.time() - start_time) * 1000, 2)  # Convert to milliseconds

            # Determine log level based on status code
            # 2xx = success, 4xx = client error, 5xx = server error
            status_code = response.status_code
            if status_code >= 500:
                log_func = logger.error
            elif status_code >= 400:
                log_func = logger.warning
            else:
                log_func = logger.info

            # Log the OUTGOING response
            log_func(
                f"[{request_id}] → {status_code} | "
                f"{method} {url} | "
                f"{elapsed_ms}ms"
            )

            # Add request ID to response headers
            # Clients can use this to report issues with a specific request
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{elapsed_ms}ms"

            return response

        except Exception as e:
            # If the route handler threw an unhandled exception
            elapsed_ms = round((time.time() - start_time) * 1000, 2)

            logger.error(
                f"[{request_id}] ✗ EXCEPTION | "
                f"{method} {url} | "
                f"{elapsed_ms}ms | "
                f"Error: {str(e)}"
            )
            raise  # Re-raise so FastAPI's error handler can handle it

    def _get_client_ip(self, request: Request) -> str:
        """
        Extract the client's real IP address from the request.

        WHY IS THIS COMPLEX?
            In production, requests often go through:
            - Load balancers
            - Reverse proxies (Nginx, Cloudflare)
            - CDNs

            These add "X-Forwarded-For" or "X-Real-IP" headers
            with the original client's IP. We check these first.

        Args:
            request: The incoming HTTP request

        Returns:
            Client IP address string
        """
        # Check proxy headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # X-Forwarded-For can be comma-separated: "client, proxy1, proxy2"
            # The FIRST IP is the original client
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct connection IP
        if request.client:
            return request.client.host

        return "unknown"
