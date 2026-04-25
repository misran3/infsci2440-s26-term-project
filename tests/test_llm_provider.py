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


def test_get_model_returns_bedrock_by_default():
    """get_model returns BedrockConverseModel when provider is bedrock."""
    from pydantic_ai.models.bedrock import BedrockConverseModel

    from src.llm.provider import get_model

    with patch.dict(os.environ, {"LLM_PROVIDER": "bedrock"}):
        model = get_model()
    assert isinstance(model, BedrockConverseModel)


def test_get_model_returns_openai():
    """get_model returns OpenAIModel when provider is openai."""
    from pydantic_ai.models.openai import OpenAIModel

    from src.llm.provider import get_model

    with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}):
        model = get_model()
    assert isinstance(model, OpenAIModel)


def test_get_model_uses_bedrock_model_env_var():
    """get_model uses BEDROCK_MODEL env var."""
    from src.llm.provider import get_model

    with patch.dict(
        os.environ,
        {"LLM_PROVIDER": "bedrock", "BEDROCK_MODEL": "us.anthropic.claude-haiku-4-5-20251001-v1:0"},
    ):
        model = get_model()
    assert "claude-haiku" in model.model_name


def test_get_model_uses_openai_model_env_var():
    """get_model uses OPENAI_MODEL env var."""
    from src.llm.provider import get_model

    with patch.dict(
        os.environ,
        {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test", "OPENAI_MODEL": "gpt-4o"},
    ):
        model = get_model()
    assert model.model_name == "gpt-4o"


from pydantic import BaseModel


class DummyOutput(BaseModel):
    """Test output model."""

    value: str


def test_get_agent_returns_agent_when_credentials_valid():
    """get_agent returns Agent when credentials are valid."""
    from pydantic_ai import Agent

    from src.llm.provider import get_agent

    with patch("src.llm.provider.validate_credentials", return_value=(True, "")):
        with patch.dict(os.environ, {"LLM_PROVIDER": "bedrock"}):
            agent = get_agent(output_type=DummyOutput, system_prompt="test")
    assert isinstance(agent, Agent)


def test_get_agent_raises_when_required_and_invalid():
    """get_agent raises LLMNotAvailableError when required=True and invalid."""
    from src.llm.provider import LLMNotAvailableError, get_agent

    with patch(
        "src.llm.provider.validate_credentials",
        return_value=(False, "Missing credentials"),
    ):
        with pytest.raises(LLMNotAvailableError, match="Missing credentials"):
            get_agent(output_type=DummyOutput, system_prompt="test", required=True)


def test_get_agent_returns_none_when_not_required_and_invalid():
    """get_agent returns None when required=False and invalid."""
    from src.llm.provider import get_agent

    with patch(
        "src.llm.provider.validate_credentials",
        return_value=(False, "Missing credentials"),
    ):
        result = get_agent(output_type=DummyOutput, system_prompt="test", required=False)
    assert result is None
