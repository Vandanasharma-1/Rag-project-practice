"""
==============================================================
app/utils/auth.py — Authentication Utilities
==============================================================

WHY THIS FILE EXISTS:
    This file handles everything authentication-related:
    - Hashing passwords (so we never store plain-text passwords)
    - Verifying passwords (checking if typed password matches stored hash)
    - Creating JWT tokens (digital "ID cards" for logged-in users)
    - Decoding JWT tokens (reading who the token belongs to)

WHAT IS PASSWORD HASHING?
    When a user registers with password "mypassword123", we DON'T
    store "mypassword123" in the database. Instead we store something
    like "$2b$12$X8lH3qGfHkF...". This is a one-way hash.

    When they log in, we hash their typed password and compare
    the TWO HASHES. If they match, password is correct.

    WHY? If your database is hacked, attackers only get
    useless hashes — not real passwords.

WHAT IS A JWT TOKEN?
    JWT = JSON Web Token
    It's a compact, digitally signed string with 3 parts:
    [HEADER].[PAYLOAD].[SIGNATURE]

    Example: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJqb2huIn0.SflKxwR...

    - HEADER: Algorithm used (HS256)
    - PAYLOAD: Data (user email, expiry time)
    - SIGNATURE: Cryptographic proof it wasn't tampered with

    After login, the server gives the user this token.
    For every subsequent request, the user sends this token.
    The server verifies it and knows WHO is making the request.

    Think of it like a concert wristband:
    - You buy a ticket (login)
    - Staff gives you a wristband (JWT token)
    - You show the wristband to enter (send token with requests)
    - Security checks it's real (server verifies signature)

HOW IT CONNECTS:
    auth.py → used by routers/auth_router.py (login/register)
    auth.py → used by routers/documents_router.py (protected routes)
    auth.py → used by routers/chat_router.py (protected routes)
==============================================================
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config.config import settings
from app.utils.logger import logger


# ==============================================================
# PASSWORD HASHING SETUP
# ==============================================================

# CryptContext sets up the password hashing algorithm.
# "bcrypt" is the industry standard — it's designed to be SLOW
# (which is GOOD! Makes brute-force attacks harder).
# deprecated="auto" means old hashes get auto-upgraded.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain_password: str) -> str:
    """
    Convert a plain-text password into a secure hash.

    HOW IT WORKS:
        1. bcrypt adds a random "salt" to your password
        2. Runs it through a complex algorithm 2^12 times
        3. Returns a 60-character hash string

    The "salt" means even if two users have the SAME password,
    their hashes will be DIFFERENT. This prevents "rainbow table" attacks.

    Args:
        plain_password: The raw password the user typed

    Returns:
        A bcrypt hash string like "$2b$12$X8lH3qGfHkFj..."

    Example:
        hash = hash_password("mypassword123")
        # hash = "$2b$12$X8lH3qGfHkFjPq..."  ← totally different from original
    """
    hashed = pwd_context.hash(plain_password)
    logger.debug("Password hashed successfully")
    return hashed


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Check if a plain-text password matches a stored hash.

    HOW IT WORKS:
        bcrypt hashes the plain_password the same way it was
        originally hashed, then compares the two hashes.

    Args:
        plain_password:   Password the user just typed
        hashed_password:  Hash stored in our database

    Returns:
        True if password is correct, False otherwise

    Example:
        stored_hash = "$2b$12$X8lH3qGfHkFjPq..."
        is_valid = verify_password("mypassword123", stored_hash)
        # is_valid = True ✓
        is_valid = verify_password("wrongpassword", stored_hash)
        # is_valid = False ✗
    """
    return pwd_context.verify(plain_password, hashed_password)


# ==============================================================
# JWT TOKEN OPERATIONS
# ==============================================================

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token for an authenticated user.

    HOW IT WORKS:
        1. Start with a copy of `data` (usually {"sub": "user@email.com"})
        2. Add an expiration time to the data
        3. Sign the data with our SECRET KEY using HS256 algorithm
        4. Return the compact JWT string

    WHAT IS "sub"?
        "sub" stands for "subject" — it's the standard JWT field
        that identifies WHO this token belongs to. We use email.

    Args:
        data:          Dictionary of claims to include in the token
                       Usually: {"sub": "user@email.com"}
        expires_delta: How long until the token expires
                       If None, uses settings.access_token_expire_minutes

    Returns:
        A JWT token string the client will store and send with requests

    Example:
        token = create_access_token({"sub": "john@example.com"})
        # token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJqb2huQGV4..."
    """
    # Make a copy so we don't modify the original dict
    to_encode = data.copy()

    # Set expiration time
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.access_token_expire_minutes
        )

    # Add expiration ("exp") to the token payload
    # JWT libraries automatically check this when decoding
    to_encode.update({"exp": expire})

    # Sign and encode the token
    # jose.jwt.encode() does 3 things:
    # 1. Base64-encodes the header and payload
    # 2. Creates a signature using our secret key
    # 3. Combines them into "header.payload.signature"
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm
    )

    logger.debug(f"Access token created, expires: {expire}")
    return encoded_jwt


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and verify a JWT token.

    HOW IT WORKS:
        1. Split the token into header, payload, signature
        2. Re-compute the signature using our secret key
        3. If signatures match → token is authentic
        4. Check if the token has expired
        5. Return the payload data

    WHAT IF THE TOKEN IS TAMPERED WITH?
        If someone changes even ONE character of the token,
        the signature verification will FAIL and we raise an error.
        This is the security guarantee of JWT.

    Args:
        token: The JWT token string from the request header

    Returns:
        The decoded payload dict, or None if invalid/expired

    Example:
        payload = decode_access_token("eyJhbGci...")
        # payload = {"sub": "john@example.com", "exp": 1234567890}
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm]
        )
        logger.debug(f"Token decoded successfully for subject: {payload.get('sub')}")
        return payload

    except JWTError as e:
        # This catches: expired tokens, invalid signatures, malformed tokens
        logger.warning(f"JWT decode failed: {e}")
        return None


def extract_username_from_token(token: str) -> Optional[str]:
    """
    Helper to extract just the username (email) from a token.

    This is a convenience wrapper around decode_access_token().
    We use this in FastAPI's dependency injection to get
    the current user's identity from their request token.

    Args:
        token: JWT token string

    Returns:
        The user's email/username, or None if token is invalid

    Example:
        username = extract_username_from_token("eyJhbGci...")
        # username = "john@example.com"
    """
    payload = decode_access_token(token)
    if payload is None:
        return None
    # "sub" is the standard JWT claim for the subject (our user email)
    return payload.get("sub")
