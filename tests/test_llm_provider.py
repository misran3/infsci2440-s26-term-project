"""Unit tests for LLM provider."""

import os
from unittest.mock import patch

import pytest


def test_get_provider_defaults_to_bedrock():
    """get_provider returns BEDROCK when env var not set."""
    from src.llm.provider import LLMProvider, get_provider

    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("LLM_PROVIDER", None)
        result = get_provider()
    assert result == LLMProvider.BEDROCK


def test_get_provider_returns_openai():
    """get_provider returns OPENAI when env var set."""
    from src.llm.provider import LLMProvider, get_provider

    with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}):
        result = get_provider()
    assert result == LLMProvider.OPENAI


def test_get_provider_invalid_raises():
    """get_provider raises ValueError for invalid provider."""
    from src.llm.provider import get_provider

    with patch.dict(os.environ, {"LLM_PROVIDER": "invalid"}):
        with pytest.raises(ValueError, match="Invalid LLM_PROVIDER"):
            get_provider()


def test_validate_credentials_openai_missing():
    """validate_credentials returns error when OPENAI_API_KEY missing."""
    from src.llm.provider import LLMProvider, validate_credentials

    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("OPENAI_API_KEY", None)
        valid, msg = validate_credentials(LLMProvider.OPENAI)
    assert valid is False
    assert "OPENAI_API_KEY" in msg


def test_validate_credentials_openai_present():
    """validate_credentials returns success when OPENAI_API_KEY set."""
    from src.llm.provider import LLMProvider, validate_credentials

    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        valid, msg = validate_credentials(LLMProvider.OPENAI)
    assert valid is True
    assert msg == ""


def test_validate_credentials_bedrock_missing():
    """validate_credentials returns error when AWS credentials missing."""
    from src.llm.provider import LLMProvider, validate_credentials

    with patch("src.llm.provider._check_bedrock_credentials", return_value=False):
        valid, msg = validate_credentials(LLMProvider.BEDROCK)
    assert valid is False
    assert "AWS" in msg


def test_validate_credentials_bedrock_present():
    """validate_credentials returns success when AWS credentials valid."""
    from src.llm.provider import LLMProvider, validate_credentials

    with patch("src.llm.provider._check_bedrock_credentials", return_value=True):
        valid, msg = validate_credentials(LLMProvider.BEDROCK)
    assert valid is True
    assert msg == ""
