"""
==============================================================
app/routers/auth_router.py — Authentication Routes
==============================================================

WHY THIS FILE EXISTS:
    This file defines the authentication API endpoints:
    - POST /auth/register → Create a new user account
    - POST /auth/login    → Login and get a JWT token
    - GET  /auth/me       → Get current user info (protected)

    It's a "router" because it handles a specific slice of the
    API functionality. Routers are registered in main.py.

WHAT IS A ROUTER?
    In FastAPI, an APIRouter groups related endpoints together.
    Instead of defining ALL routes in main.py (which would become
    huge), we split them into logical routers:
    - auth_router.py   → /auth/* routes
    - documents_router → /documents/* routes
    - chat_router      → /chat/* routes

    In main.py, we just do:
        app.include_router(auth_router, prefix="/api/v1/auth")

IMPORTANT: IN-MEMORY USER STORE
    For simplicity, this demo stores users in a Python dict.
    In production, you'd use a real database (PostgreSQL, MongoDB).

    In-memory store means: data is lost when the server restarts.

AUTHENTICATION FLOW:
    1. Client POST /auth/register → Server creates user, hashes password
    2. Client POST /auth/login → Server verifies password, returns JWT
    3. Client stores JWT (in localStorage or cookies)
    4. Client sends JWT in header: "Authorization: Bearer <token>"
    5. Server verifies JWT on protected routes using Depends()

FASTAPI DEPENDENCY INJECTION WITH Depends():
    Protected routes use a "dependency" that runs BEFORE the
    route handler. The dependency verifies the JWT token.

    @router.get("/me")
    async def get_me(current_user = Depends(get_current_user)):
        # current_user is automatically populated from the token
        # If token is invalid/missing, this route returns 401

    This is called "Dependency Injection" — FastAPI "injects"
    the current_user value into the route function automatically.

HOW IT CONNECTS:
    auth_router.py → uses auth.py (password hashing, JWT)
    auth_router.py → registered in main.py
    auth_router.py → uses schemas.py (request/response models)
==============================================================
"""

from datetime import timedelta
from typing import Dict

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from app.models.schemas import (
    UserRegisterRequest,
    UserLoginRequest,
    TokenResponse,
    UserResponse,
    ErrorResponse
)
from app.utils.auth import hash_password, verify_password, create_access_token, extract_username_from_token
from app.utils.logger import logger
from app.config.config import settings

# ==============================================================
# ROUTER SETUP
# ==============================================================

# Create a router with a tag (shows in Swagger UI)
router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],       # Groups endpoints in /docs UI
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        422: {"description": "Validation Error"}
    }
)

# ==============================================================
# IN-MEMORY USER DATABASE
# ==============================================================
# ⚠️ DEMO ONLY: In production, use a real database!
# This is a simple dictionary: {email: user_data}
#
# Example structure:
# users_db = {
#     "john@example.com": {
#         "email": "john@example.com",
#         "full_name": "John Doe",
#         "hashed_password": "$2b$12$...",
#         "is_active": True,
#         "created_at": datetime(...)
#     }
# }
users_db: Dict[str, dict] = {}

# ==============================================================
# OAUTH2 SCHEME
# ==============================================================
# OAuth2PasswordBearer is a FastAPI utility that:
# 1. Looks for the "Authorization: Bearer <token>" header
# 2. Extracts the token
# 3. Returns it to the dependency function
#
# tokenUrl tells Swagger UI where the login endpoint is
# (for the Swagger "Authorize" button)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


# ==============================================================
# DEPENDENCY: Get Current User
# ==============================================================

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    FastAPI dependency that validates JWT and returns the current user.

    WHAT IS A DEPENDENCY?
        A function decorated with Depends() is called automatically
        before the route handler. It can:
        - Validate inputs
        - Check authentication
        - Connect to databases
        - Inject objects into route functions

    THIS DEPENDENCY DOES:
        1. Extracts the JWT token from the Authorization header
        2. Decodes and validates the JWT
        3. Finds the user in our database
        4. Returns the user dict (or raises 401 if invalid)

    USAGE IN ROUTES:
        @router.get("/profile")
        async def get_profile(user = Depends(get_current_user)):
            # user is automatically the logged-in user's data
            return {"email": user["email"]}

    Args:
        token: JWT token extracted by OAuth2PasswordBearer

    Returns:
        User dict for the authenticated user

    Raises:
        HTTPException 401: If token is invalid, expired, or user not found
    """
    # Define the 401 exception we'll raise if anything fails
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials. Please login again.",
        headers={"WWW-Authenticate": "Bearer"},  # Standard OAuth2 header
    )

    # Step 1: Extract username from token
    username = extract_username_from_token(token)
    if username is None:
        logger.warning("Token validation failed: invalid or expired token")
        raise credentials_exception

    # Step 2: Find user in our "database"
    user = users_db.get(username)
    if user is None:
        logger.warning(f"Token validation failed: user not found: {username}")
        raise credentials_exception

    # Step 3: Check if user is active
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has been deactivated."
        )

    return user


# ==============================================================
# API ROUTES
# ==============================================================

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="""
    Create a new user account.

    **Requirements:**
    - Email must be valid and unique
    - Password must be at least 8 characters with 1 letter and 1 digit
    - Full name must be 2-100 characters

    **Example:**
    ```json
    {
        "email": "john@example.com",
        "password": "SecurePass123",
        "full_name": "John Doe"
    }
    ```
    """
)
async def register(user_data: UserRegisterRequest):
    """
    Register a new user account.

    FLOW:
        1. Validate request data (Pydantic does this automatically)
        2. Check email isn't already registered
        3. Hash the password (NEVER store plain text!)
        4. Save user to database
        5. Return user info (without password)

    Args:
        user_data: Registration data (email, password, full_name)

    Returns:
        UserResponse with the created user's info

    Raises:
        HTTPException 409: If email already registered
    """
    email = user_data.email.lower()  # Normalize email to lowercase

    logger.info(f"Registration attempt for: {email}")

    # Check if email already exists
    if email in users_db:
        logger.warning(f"Registration failed: email already exists: {email}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"An account with email '{email}' already exists. Please login instead."
        )

    # Hash the password
    # CRITICAL: We NEVER store the plain-text password!
    hashed_pw = hash_password(user_data.password)

    # Store user in our "database"
    from datetime import datetime, timezone
    user_record = {
        "email": email,
        "full_name": user_data.full_name,
        "hashed_password": hashed_pw,
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
    }
    users_db[email] = user_record

    logger.info(f"User registered successfully: {email}")

    # Return user info (password excluded)
    return UserResponse(
        email=user_record["email"],
        full_name=user_record["full_name"],
        created_at=user_record["created_at"],
        is_active=True
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and get JWT token",
    description="""
    Authenticate with email and password to receive a JWT access token.

    **The returned token should be:**
    - Stored securely by the client
    - Sent in the `Authorization: Bearer <token>` header for all authenticated requests

    **Token expires in:** `ACCESS_TOKEN_EXPIRE_MINUTES` (default: 60 minutes)
    """
)
async def login(login_data: UserLoginRequest):
    """
    Authenticate user and return JWT token.

    AUTHENTICATION FLOW:
        1. Look up user by email
        2. Verify password against stored hash
        3. Create JWT token containing user's email
        4. Return token + metadata

    SECURITY NOTE:
        We always take the same amount of time whether the user exists
        or not. This prevents "timing attacks" where an attacker can
        determine if an email is registered based on response time.

    Args:
        login_data: Login credentials (email, password)

    Returns:
        TokenResponse with JWT token

    Raises:
        HTTPException 401: If credentials are incorrect
    """
    email = login_data.email.lower()
    logger.info(f"Login attempt for: {email}")

    # Generic error message — don't reveal if email exists!
    # Specific messages like "email not found" help attackers
    auth_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid email or password. Please check your credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Look up user
    user = users_db.get(email)
    if user is None:
        logger.warning(f"Login failed: user not found: {email}")
        raise auth_error

    # Verify password
    if not verify_password(login_data.password, user["hashed_password"]):
        logger.warning(f"Login failed: incorrect password for: {email}")
        raise auth_error

    # Check account is active
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has been deactivated."
        )

    # Create JWT token
    # The "sub" (subject) field holds the user's identity
    access_token = create_access_token(
        data={"sub": email},
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes)
    )

    logger.info(f"Login successful for: {email}")

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,  # Convert to seconds
        user_email=email
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user info",
    description="Returns the profile of the currently authenticated user. Requires valid JWT token."
)
async def get_me(current_user: dict = Depends(get_current_user)):
    """
    Return the current authenticated user's profile.

    This is a PROTECTED ROUTE — it requires a valid JWT token.

    How Depends(get_current_user) works:
        1. FastAPI sees Depends(get_current_user)
        2. Calls get_current_user() automatically BEFORE this function
        3. get_current_user() reads Authorization header, validates JWT
        4. If valid: passes user dict to current_user parameter
        5. If invalid: raises 401 HTTPException (this function never runs)

    Args:
        current_user: Injected by Depends() — the authenticated user dict

    Returns:
        UserResponse with user info
    """
    return UserResponse(
        email=current_user["email"],
        full_name=current_user["full_name"],
        created_at=current_user["created_at"],
        is_active=current_user["is_active"]
    )


@router.post(
    "/logout",
    summary="Logout (client-side)",
    description="""
    Logout the current user.

    **Note:** Since JWTs are stateless, true server-side logout requires
    a token blacklist (Redis). For this demo, logout is handled client-side
    by deleting the stored token.
    """
)
async def logout(current_user: dict = Depends(get_current_user)):
    """
    Logout endpoint (informational — actual logout is client-side).

    WITH JWT, REAL LOGOUT IS COMPLEX:
        JWT tokens are self-contained and valid until they expire.
        To truly invalidate a token server-side, you'd need:
        1. A blacklist database (like Redis)
        2. Check the blacklist on every request

        For this demo, we just return a success message.
        The client should delete the stored token.

    Args:
        current_user: The authenticated user (from JWT)

    Returns:
        Success message
    """
    logger.info(f"Logout for user: {current_user['email']}")
    return {
        "message": "Logged out successfully. Please delete your stored token.",
        "user": current_user["email"]
    }
