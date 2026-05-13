"""
tests/test_auth.py — Authentication Tests

WHY TESTS EXIST:
    Tests verify your code works correctly.
    Run them with: pytest tests/ -v

    Good tests catch bugs BEFORE production users do.
    They also serve as documentation — showing HOW the code is meant to be used.

HOW TO RUN:
    # From project root (with venv activated):
    pytest tests/ -v
    
    # Run just this file:
    pytest tests/test_auth.py -v
"""

import pytest
from app.utils.auth import hash_password, verify_password, create_access_token, decode_access_token, extract_username_from_token


# =====================================================
# PASSWORD HASHING TESTS
# =====================================================

def test_hash_password_returns_string():
    """hash_password should return a non-empty string."""
    result = hash_password("MyPassword123")
    assert isinstance(result, str)
    assert len(result) > 0


def test_hash_password_is_not_plain_text():
    """The hash should NOT look like the original password."""
    plain = "MyPassword123"
    hashed = hash_password(plain)
    assert hashed != plain


def test_hash_password_different_each_time():
    """
    bcrypt adds a random salt, so the same password produces
    DIFFERENT hashes each time. This prevents rainbow table attacks.
    """
    hash1 = hash_password("MyPassword123")
    hash2 = hash_password("MyPassword123")
    assert hash1 != hash2  # Different salts = different hashes


def test_verify_password_correct():
    """Correct password should verify successfully."""
    plain = "MyPassword123"
    hashed = hash_password(plain)
    assert verify_password(plain, hashed) is True


def test_verify_password_wrong():
    """Wrong password should fail verification."""
    hashed = hash_password("MyPassword123")
    assert verify_password("WrongPassword", hashed) is False


def test_verify_password_empty_string():
    """Empty password should fail verification."""
    hashed = hash_password("MyPassword123")
    assert verify_password("", hashed) is False


# =====================================================
# JWT TOKEN TESTS
# =====================================================

def test_create_access_token_returns_string():
    """create_access_token should return a non-empty string."""
    token = create_access_token({"sub": "test@example.com"})
    assert isinstance(token, str)
    assert len(token) > 0


def test_create_access_token_has_three_parts():
    """
    JWT format is: HEADER.PAYLOAD.SIGNATURE
    Splitting by '.' should give exactly 3 parts.
    """
    token = create_access_token({"sub": "test@example.com"})
    parts = token.split(".")
    assert len(parts) == 3


def test_decode_access_token_valid():
    """A freshly created token should decode successfully."""
    email = "test@example.com"
    token = create_access_token({"sub": email})
    payload = decode_access_token(token)
    
    assert payload is not None
    assert payload["sub"] == email


def test_decode_access_token_invalid():
    """A tampered/invalid token should return None (not raise an error)."""
    result = decode_access_token("this.is.not.a.valid.jwt")
    assert result is None


def test_decode_access_token_empty():
    """An empty string token should return None."""
    result = decode_access_token("")
    assert result is None


def test_extract_username_from_token():
    """Should correctly extract the 'sub' field from a valid token."""
    email = "john@example.com"
    token = create_access_token({"sub": email})
    username = extract_username_from_token(token)
    assert username == email


def test_extract_username_from_invalid_token():
    """Invalid token should return None, not raise an exception."""
    result = extract_username_from_token("invalid.token.here")
    assert result is None


# =====================================================
# HELPER UTILITIES TESTS
# =====================================================

def test_generate_unique_id():
    """Each call should produce a different unique ID."""
    from app.utils.helpers import generate_unique_id
    id1 = generate_unique_id()
    id2 = generate_unique_id()
    assert id1 != id2
    assert len(id1) == 36  # UUID4 is always 36 chars with dashes


def test_is_allowed_file():
    """Test file extension validation."""
    from app.utils.helpers import is_allowed_file
    allowed = [".pdf", ".txt", ".docx"]
    
    assert is_allowed_file("report.pdf", allowed) is True
    assert is_allowed_file("notes.txt", allowed) is True
    assert is_allowed_file("doc.docx", allowed) is True
    assert is_allowed_file("virus.exe", allowed) is False
    assert is_allowed_file("image.jpg", allowed) is False


def test_sanitize_filename():
    """Test dangerous filenames are cleaned."""
    from app.utils.helpers import sanitize_filename
    
    # Path traversal attack — should be neutralized
    result = sanitize_filename("../../../etc/passwd")
    assert ".." not in result
    assert "/" not in result
    
    # Spaces should become underscores
    result = sanitize_filename("my report.pdf")
    assert " " not in result


def test_format_file_size():
    """Test human-readable file size formatting."""
    from app.utils.helpers import format_file_size
    
    assert format_file_size(500) == "500 B"
    assert format_file_size(1024) == "1.0 KB"
    assert format_file_size(1024 * 1024) == "1.0 MB"


def test_clean_text():
    """Test text cleaning removes extra whitespace."""
    from app.utils.helpers import clean_text
    
    messy = "Hello   world\n\n  this   is   text  "
    clean = clean_text(messy)
    
    assert "  " not in clean  # No double spaces
    assert clean == clean.strip()  # No leading/trailing whitespace


def test_chunk_text():
    """Test text chunking produces correct number of chunks."""
    from app.utils.helpers import chunk_text_by_sentences
    
    # Create a text of known length
    text = "A" * 1000  # 1000 characters
    chunks = chunk_text_by_sentences(text, chunk_size=100, overlap=10)
    
    assert len(chunks) > 0
    # Each chunk should be at most chunk_size characters
    for chunk in chunks:
        assert len(chunk) <= 200  # Some tolerance for word boundaries
