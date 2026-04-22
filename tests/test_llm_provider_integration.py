"""Integration tests for LLM provider with real credentials.

These tests require actual credentials to be set:
- AWS_PROFILE and AWS_REGION for Bedrock
- OPENAI_API_KEY for OpenAI

Skip if credentials are not available.
"""

import os

import pytest
from pydantic import BaseModel

from src.llm.provider import LLMProvider, get_agent, get_provider, validate_credentials


class SimpleOutput(BaseModel):
    """Simple output for testing."""

    answer: str


@pytest.fixture
def has_bedrock_credentials() -> bool:
    """Check if Bedrock credentials are available."""
    valid, _ = validate_credentials(LLMProvider.BEDROCK)
    return valid


@pytest.fixture
def has_openai_credentials() -> bool:
    """Check if OpenAI credentials are available."""
    return bool(os.getenv("OPENAI_API_KEY"))


@pytest.mark.asyncio
async def test_bedrock_agent_runs(has_bedrock_credentials):
    """Test that Bedrock agent can run a simple query."""
    if not has_bedrock_credentials:
        pytest.skip("Bedrock credentials not available")

    os.environ["LLM_PROVIDER"] = "bedrock"
    agent = get_agent(
        output_type=SimpleOutput,
        system_prompt="Answer with a single word.",
    )
    assert agent is not None

    result = await agent.run("What is 1+1? Answer with just the number.")
    assert result.output.answer is not None


@pytest.mark.asyncio
async def test_openai_agent_runs(has_openai_credentials):
    """Test that OpenAI agent can run a simple query."""
    if not has_openai_credentials:
        pytest.skip("OpenAI credentials not available")

    os.environ["LLM_PROVIDER"] = "openai"
    agent = get_agent(
        output_type=SimpleOutput,
        system_prompt="Answer with a single word.",
    )
    assert agent is not None

    result = await agent.run("What is 1+1? Answer with just the number.")
    assert result.output.answer is not None
