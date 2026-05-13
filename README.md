# 🤖 Enterprise GenAI Assistant & RAG Pipeline

A **production-grade** backend for document-based question answering using **Retrieval-Augmented Generation (RAG)**. Upload your enterprise documents (PDF, DOCX, TXT) and ask natural language questions — powered by Google Gemini AI and ChromaDB vector database.

---

## 📋 Table of Contents

1. [What Is This Project?](#what-is-this-project)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Setup Guide (Step by Step)](#setup-guide-step-by-step)
6. [Getting Your Gemini API Key](#getting-your-gemini-api-key)
7. [Running the Project](#running-the-project)
8. [API Endpoints](#api-endpoints)
9. [Using Swagger UI](#using-swagger-ui)
10. [Example curl Requests](#example-curl-requests)
11. [Troubleshooting](#troubleshooting)
12. [Production Deployment Notes](#production-deployment-notes)

---

## What Is This Project?

This is a **RAG (Retrieval-Augmented Generation)** system. Here's what that means in plain English:

**Without RAG:**
> You: "What was our Q3 revenue?"
> Generic AI: "I don't have access to your company data."

**With RAG (this project):**
> You upload your Q3 financial report PDF.
> You: "What was our Q3 revenue?"
> Our AI: "Based on the Q3 Financial Report, revenue was $4.2 billion, representing a 12% year-over-year increase, primarily driven by the Asia-Pacific region."

The AI answers **from YOUR documents**, not from generic training data.

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Web Framework | FastAPI | API routes, request handling |
| ASGI Server | Uvicorn | Runs the FastAPI app |
| AI Model | Google Gemini 1.5 Flash | Generates natural language answers |
| Vector Database | ChromaDB | Stores and searches embeddings |
| Embeddings | SentenceTransformers | Converts text to numerical vectors |
| Authentication | JWT + bcrypt | Secure user login |
| Data Validation | Pydantic | Request/response schemas |
| Logging | Loguru | Beautiful, structured logging |
| Config | python-dotenv | Environment variable management |

---

## Project Structure

```
enterprise-rag-assistant/
│
├── app/                          # Main application package
│   ├── main.py                   # FastAPI app + middleware + routers
│   │
│   ├── config/
│   │   └── config.py             # All settings (reads from .env)
│   │
│   ├── models/
│   │   └── schemas.py            # Pydantic request/response models
│   │
│   ├── routers/                  # API endpoint definitions
│   │   ├── auth_router.py        # /auth/* (register, login)
│   │   ├── documents_router.py   # /documents/* (upload, list, delete)
│   │   └── chat_router.py        # /chat/* (ask questions)
│   │
│   ├── services/                 # Business logic layer
│   │   ├── document_processor.py # Extract text from PDF/TXT/DOCX
│   │   ├── vector_store.py       # Embeddings + ChromaDB operations
│   │   ├── llm_client.py         # Google Gemini API client
│   │   └── rag_pipeline.py       # Orchestrates the full RAG flow
│   │
│   ├── utils/
│   │   ├── auth.py               # Password hashing, JWT tokens
│   │   ├── logger.py             # Loguru logging setup
│   │   └── helpers.py            # Utility functions
│   │
│   ├── middleware/
│   │   └── request_logger.py     # HTTP request/response logging
│   │
│   └── database/
│       └── chroma_manager.py     # ChromaDB connection manager
│
├── data/
│   ├── uploads/                  # Uploaded documents stored here
│   ├── chroma_db/                # ChromaDB vector data stored here
│   └── logs/                     # Log files stored here
│
├── tests/                        # Test files (pytest)
│
├── .env.example                  # Template for environment variables
├── requirements.txt              # Python dependencies
├── run.py                        # Start the server
└── README.md                     # This file
```

---

## Prerequisites

Before starting, ensure you have:

- **Python 3.10 or higher** — [Download](https://python.org/downloads)
- **pip** — Comes with Python
- **Git** — [Download](https://git-scm.com/downloads)
- **Google account** — For Gemini API key (free)

Check your Python version:
```bash
python --version
# Should output: Python 3.10.x or higher
```

---

## Setup Guide (Step by Step)

### Step 1: Clone or Download the Project

```bash
# If using git:
git clone https://github.com/your-repo/enterprise-rag-assistant.git
cd enterprise-rag-assistant

# Or if you have the folder already:
cd enterprise-rag-assistant
```

### Step 2: Create a Virtual Environment

**What is a virtual environment?**
A virtual environment is an isolated Python installation. It keeps this project's dependencies separate from other projects. Like having a separate toolbox for each project.

```bash
# Create virtual environment named "venv"
python -m venv venv

# Activate it:

# On Windows (Command Prompt):
venv\Scripts\activate

# On Windows (PowerShell):
venv\Scripts\Activate.ps1

# On macOS/Linux:
source venv/bin/activate
```

After activation, your terminal prompt changes to show `(venv)`:
```
(venv) C:\projects\enterprise-rag-assistant>
```

> ⚠️ **Always activate the virtual environment** before working on this project!

### Step 3: Install Dependencies

```bash
# Make sure venv is activated first!
pip install -r requirements.txt
```

This installs ~30 packages. It may take 2-5 minutes, especially the first time (SentenceTransformers downloads ML models).

You should see output ending with:
```
Successfully installed fastapi-0.111.0 uvicorn-0.29.0 chromadb-0.5.0 ...
```

### Step 4: Set Up Environment Variables

```bash
# Copy the example env file
cp .env.example .env

# Open .env in your editor and fill in the values
# On Windows: notepad .env
# On macOS:   open -e .env
# On Linux:   nano .env
```

Your `.env` file should look like:
```env
GEMINI_API_KEY=AIzaSyYour-actual-api-key-here
JWT_SECRET=change-this-to-a-long-random-string-min-32-chars
ACCESS_TOKEN_EXPIRE_MINUTES=60
CHROMA_DB_PATH=./data/chroma_db
LOG_LEVEL=INFO
UPLOAD_DIR=./data/uploads
LOG_DIR=./data/logs
DEBUG=True
```

> 🔒 **Never commit your `.env` file to Git!** It contains secrets.

---

## Getting Your Gemini API Key

Google Gemini has a **generous free tier** — perfect for learning and development.

**Step-by-step:**

1. Go to **[https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)**

2. Sign in with your Google account

3. Click **"Create API Key"**

4. Select an existing Google Cloud project, or click **"Create API key in new project"**

5. Copy the API key (starts with `AIza...`)

6. Paste it into your `.env` file:
   ```
   GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXX
   ```

**Free Tier Limits (as of 2024):**
- 15 requests per minute
- 1 million tokens per day
- No credit card required

> ⚠️ Keep your API key secret! Anyone with it can make API calls charged to your account.

---

## Running the Project

### Start the Server

```bash
# Make sure venv is activated!
python run.py
```

You should see:
```
╔══════════════════════════════════════════════════════════╗
║          Enterprise RAG Assistant  v1.0.0               ║
╠══════════════════════════════════════════════════════════╣
║  Starting server...                                      ║
║                                                          ║
║  Local:    http://localhost:8000                         ║
║  API:      http://localhost:8000/api/v1                  ║
║  Docs:     http://localhost:8000/docs                    ║
║  Health:   http://localhost:8000/health                  ║
╚══════════════════════════════════════════════════════════╝

INFO | ChromaDB initialized | Collection: 'enterprise_docs' | Existing vectors: 0
INFO | Application ready! Listening for requests...
```

### Verify It's Running

Open your browser and visit:
- **http://localhost:8000** — Root endpoint (JSON info)
- **http://localhost:8000/health** — Health check
- **http://localhost:8000/docs** — Interactive Swagger UI ← **Start here!**

---

## API Endpoints

### Authentication

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/v1/auth/register` | Create new account | ❌ No |
| POST | `/api/v1/auth/login` | Login, get JWT token | ❌ No |
| GET | `/api/v1/auth/me` | Get current user info | ✅ Yes |
| POST | `/api/v1/auth/logout` | Logout | ✅ Yes |

### Documents

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/v1/documents/upload` | Upload & process document | ✅ Yes |
| GET | `/api/v1/documents/` | List all documents | ✅ Yes |
| GET | `/api/v1/documents/{id}` | Get document details | ✅ Yes |
| DELETE | `/api/v1/documents/{id}` | Delete document | ✅ Yes |
| GET | `/api/v1/documents/stats/overview` | Storage statistics | ✅ Yes |

### Chat / RAG

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/v1/chat/ask` | Ask a question (main RAG) | ✅ Yes |
| POST | `/api/v1/chat/search` | Semantic search only | ✅ Yes |
| GET | `/api/v1/chat/history` | Conversation history | ✅ Yes |
| DELETE | `/api/v1/chat/history` | Clear history | ✅ Yes |

### System

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/` | Root info | ❌ No |
| GET | `/health` | Health check | ❌ No |
| GET | `/api/v1` | API directory | ❌ No |
| GET | `/docs` | Swagger UI | ❌ No |

---

## Using Swagger UI

Swagger UI is an interactive API explorer built into FastAPI. **No Postman needed!**

**Step-by-step workflow:**

### 1. Open Swagger
Navigate to: **http://localhost:8000/docs**

### 2. Register a User
- Click **POST /api/v1/auth/register**
- Click **"Try it out"**
- Enter your details in the request body:
  ```json
  {
    "email": "you@example.com",
    "password": "SecurePass123",
    "full_name": "Your Name"
  }
  ```
- Click **"Execute"**
- You should get a `201 Created` response

### 3. Login to Get a Token
- Click **POST /api/v1/auth/login**
- Click **"Try it out"**
- Enter:
  ```json
  {
    "email": "you@example.com",
    "password": "SecurePass123"
  }
  ```
- Click **"Execute"**
- Copy the `access_token` from the response (starts with `eyJ...`)

### 4. Authorize with Your Token
- Click the **🔒 Authorize** button (top right of Swagger page)
- In the Value field, type: `Bearer eyJhbGci...` (your full token)
- Click **Authorize**, then **Close**

Now the 🔒 icon on protected endpoints turns to 🔓.

### 5. Upload a Document
- Click **POST /api/v1/documents/upload**
- Click **"Try it out"**
- Click **"Choose File"** and select a PDF, TXT, or DOCX
- Click **"Execute"**
- Note the returned `document_id`

### 6. Ask a Question
- Click **POST /api/v1/chat/ask**
- Click **"Try it out"**
- Enter:
  ```json
  {
    "question": "What are the main topics in this document?",
    "top_k": 5
  }
  ```
- Click **"Execute"**
- Read the AI's answer with source citations!

---

## Example curl Requests

**curl** is a command-line tool for making HTTP requests. Use it to test the API from your terminal.

### Register
```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "john@example.com",
    "password": "SecurePass123",
    "full_name": "John Doe"
  }'
```

Expected response:
```json
{
  "email": "john@example.com",
  "full_name": "John Doe",
  "created_at": "2024-01-15T10:30:00Z",
  "is_active": true
}
```

### Login
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "john@example.com",
    "password": "SecurePass123"
  }'
```

Expected response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user_email": "john@example.com"
}
```

Save the token:
```bash
# Save token to a variable (Linux/macOS)
TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# On Windows (PowerShell):
$TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### Upload a Document
```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/your/document.pdf"
```

Expected response:
```json
{
  "document_id": "a47f3d2c-8b9e-4f1a-b2c3-d4e5f6789abc",
  "filename": "document.pdf",
  "status": "processed",
  "chunks_created": 42,
  "file_size": "1.2 MB",
  "message": "Document processed and indexed successfully. 42 chunks created."
}
```

### Ask a Question
```bash
curl -X POST http://localhost:8000/api/v1/chat/ask \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main points in this document?",
    "top_k": 5
  }'
```

Expected response:
```json
{
  "question": "What are the main points in this document?",
  "answer": "Based on the uploaded document, the main points are...",
  "retrieved_chunks": [
    {
      "chunk_id": "a47f3d2c_chunk_0",
      "content": "The document discusses...",
      "relevance_score": 0.87,
      "document_id": "a47f3d2c-8b9e-4f1a-b2c3-d4e5f6789abc",
      "filename": "document.pdf",
      "chunk_index": 0
    }
  ],
  "processing_time_seconds": 2.341,
  "model_used": "gemini-1.5-flash",
  "chunks_retrieved": 5,
  "status": "success"
}
```

### List Documents
```bash
curl -X GET http://localhost:8000/api/v1/documents/ \
  -H "Authorization: Bearer $TOKEN"
```

### Health Check
```bash
curl http://localhost:8000/health
```

---

## Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'app'`
**Cause:** You're running Python from the wrong directory.
**Fix:** Make sure you're in the `enterprise-rag-assistant` root folder:
```bash
cd enterprise-rag-assistant
python run.py
```

### ❌ `GEMINI_API_KEY` environment variable not set
**Cause:** `.env` file is missing or has wrong content.
**Fix:**
```bash
# Check .env exists
ls -la .env

# Check its content
cat .env

# Make sure GEMINI_API_KEY is set:
# GEMINI_API_KEY=AIzaSyXXXXX...
```

### ❌ `Address already in use` (Port 8000)
**Cause:** Something else is using port 8000.
**Fix:** Kill the process or change the port:
```bash
# Find what's using port 8000:
# On Linux/macOS:
lsof -i :8000

# On Windows:
netstat -ano | findstr :8000

# Kill it or change port in run.py:
# port=8001
```

### ❌ `Failed to initialize ChromaDB`
**Cause:** Permission issue or corrupted database.
**Fix:**
```bash
# Delete and recreate ChromaDB data:
rm -rf data/chroma_db
mkdir data/chroma_db
python run.py
```

### ❌ `No text could be extracted from the PDF`
**Cause:** The PDF is an image scan (not text-based).
**Fix:** Use a PDF with actual text content, or use OCR software to convert the scanned PDF to text first.

### ❌ `Invalid API key` from Gemini
**Cause:** API key is wrong or expired.
**Fix:**
1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Generate a new API key
3. Update `.env` with the new key
4. Restart the server

### ❌ Slow first request
**Cause:** SentenceTransformers model is downloading on first use.
**Fix:** Wait for the download to complete (~22MB). Subsequent requests are fast.

### ❌ `401 Unauthorized` on protected endpoints
**Cause:** Missing or expired JWT token.
**Fix:**
1. Login again: `POST /api/v1/auth/login`
2. Copy the new `access_token`
3. Include it as: `Authorization: Bearer <token>`

### ❌ `422 Validation Error`
**Cause:** Your request body doesn't match the expected schema.
**Fix:** Check the error details in the response body. The `details` field tells you exactly which field is wrong and why.

---

## Production Deployment Notes

> ⚠️ This project is production-ready in architecture but has some demo simplifications.

### What to Change for Real Production:

1. **Replace in-memory user store** (`auth_router.py`) with a real database (PostgreSQL + SQLAlchemy)

2. **Replace in-memory document registry** (`documents_router.py`) with a real database

3. **Set strong secrets:**
   ```env
   JWT_SECRET=<64-character-random-string>
   DEBUG=False
   ```

4. **Restrict CORS origins:**
   ```env
   # In config.py, change:
   cors_origins: list[str] = ["https://yourapp.com"]
   ```

5. **Use multiple Uvicorn workers:**
   ```bash
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app
   ```

6. **Add HTTPS/SSL** using Nginx as a reverse proxy

7. **Set up proper log aggregation** (CloudWatch, Datadog, ELK Stack)

8. **Add rate limiting** to prevent API abuse

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
start htmlcov/index.html # Windows
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m 'Add my feature'`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## License

MIT License — Free for personal and commercial use.
