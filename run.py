"""
==============================================================
run.py — Application Server Entry Point
==============================================================

WHY THIS FILE EXISTS:
    This is the file you run to START the server.
    It configures and launches Uvicorn, which is the ASGI server
    that runs our FastAPI application.

    Instead of remembering complex command-line flags, you just run:
        python run.py

WHAT IS UVICORN?
    Uvicorn is an ASGI (Asynchronous Server Gateway Interface) server.
    Think of it as the "engine" that:
    1. Opens a port (8000) on your machine
    2. Listens for incoming HTTP requests
    3. Passes each request to FastAPI for processing
    4. Returns FastAPI's response back to the client

    Without Uvicorn, your FastAPI code is just functions — Uvicorn
    makes them accessible over the network.

UVICORN VS GUNICORN:
    - Uvicorn: Single-process async server (great for development)
    - Gunicorn: Multi-process server with Uvicorn workers (production)
    
    For learning/development: `python run.py` (Uvicorn directly)
    For production: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app`

WHAT IS --reload?
    In development mode (debug=True), Uvicorn watches your Python files.
    If you change any .py file, it automatically restarts the server.
    You don't have to manually stop and restart after every code change!

HOW IT CONNECTS:
    run.py → starts Uvicorn
    Uvicorn → runs app.main:app (our FastAPI application)
    FastAPI processes all requests from there
==============================================================
"""

import uvicorn
from app.config.config import settings


def main():
    """
    Configure and start the Uvicorn ASGI server.
    
    CONFIGURATION BREAKDOWN:
    
    "app.main:app"
        This tells Uvicorn WHERE to find the FastAPI application.
        Format: "module_path:variable_name"
        - "app.main"  → look in the app/main.py file
        - ":app"      → use the variable named `app`
        
        In app/main.py we have: app = FastAPI(...)
        So Uvicorn finds that FastAPI instance.
    
    host="0.0.0.0"
        Listen on ALL network interfaces:
        - "127.0.0.1" (localhost) → only your own machine can connect
        - "0.0.0.0"               → any machine on the network can connect
        
        Use "0.0.0.0" for development (so your phone/other devices can test).
        In production, use a reverse proxy (Nginx) in front.
    
    port=8000
        The port number to listen on.
        http://localhost:8000 is the URL to access the API.
        You can change this if 8000 is already in use.
    
    reload=settings.debug
        - True (debug mode): Auto-restart on file changes
        - False (production): No auto-restart (more stable)
        
        Set DEBUG=True in .env for development.
        Set DEBUG=False in .env for production.
    
    workers=1
        Number of worker processes.
        - Development: 1 worker is fine
        - Production: Use (2 × CPU cores) + 1 workers
          For 4-core machine: 9 workers
        
        NOTE: reload=True forces workers=1 (can't multi-process with reload)
    
    log_level
        Uvicorn's own log level (separate from our Loguru logger).
        We set it to match our application's log level.
    
    access_log=False
        We have our own RequestLoggingMiddleware, so we disable
        Uvicorn's built-in access log to avoid duplicate log lines.
    """

    print(f"""
╔══════════════════════════════════════════════════════════╗
║          Enterprise RAG Assistant  v{settings.app_version}              ║
╠══════════════════════════════════════════════════════════╣
║  Starting server...                                      ║
║                                                          ║
║  Local:    http://localhost:8000                         ║
║  API:      http://localhost:8000/api/v1                  ║
║  Docs:     http://localhost:8000/docs                    ║
║  Health:   http://localhost:8000/health                  ║
║                                                          ║
║  Debug mode: {str(settings.debug):<46}  ║
║  Log level:  {settings.log_level:<46}  ║
╚══════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        # The FastAPI app to run
        # "app.main:app" means: file=app/main.py, variable=app
        app="app.main:app",

        # Network settings
        host="0.0.0.0",    # Listen on all interfaces
        port=8000,          # Port number

        # Development mode: auto-reload on file changes
        # Production mode: set DEBUG=False in .env
        reload=settings.debug,

        # Workers (only 1 when reload=True)
        workers=1,

        # Log settings
        log_level=settings.log_level.lower(),

        # Disable Uvicorn's access log — we use RequestLoggingMiddleware
        access_log=False,

        # SSL/TLS (for HTTPS) - uncomment for production with SSL:
        # ssl_keyfile="./certs/key.pem",
        # ssl_certfile="./certs/cert.pem",

        # Server headers
        server_header=False,  # Don't expose server info (security)
        date_header=True,

        # Proxy headers (important when behind Nginx/load balancer)
        # forwarded_allow_ips="*",  # Trust X-Forwarded-For from any proxy

        # Timeouts
        timeout_keep_alive=30,  # Keep-alive connection timeout in seconds
    )


if __name__ == "__main__":
    """
    This block runs when you execute: python run.py
    
    `if __name__ == "__main__"` is a Python idiom meaning:
    "Only run this code if this file is executed directly,
    not if it's imported by another module."
    
    So `python run.py` → runs main()
    But `import run` → does NOT run main()
    """
    main()
