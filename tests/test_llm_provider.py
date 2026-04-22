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
