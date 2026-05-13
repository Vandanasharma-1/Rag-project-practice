"""
tests/__init__.py

Makes `tests` a Python package so pytest can discover test files.

HOW TO RUN TESTS:
    # Run all tests
    pytest tests/

    # Run with verbose output
    pytest tests/ -v

    # Run a specific test file
    pytest tests/test_auth.py

    # Run with coverage report
    pytest tests/ --cov=app --cov-report=html
"""
