"""Reusable LLM provider using Pydantic AI with multi-backend support."""

import os
from enum import Enum

from pydantic_ai import Agent


class LLMProvider(Enum):
    """Supported LLM providers."""

    BEDROCK = "bedrock"
    OPENAI = "openai"


class LLMNotAvailableError(Exception):
    """Raised when LLM is required but credentials are missing."""

    pass


def get_provider() -> LLMProvider:
    """Read LLM_PROVIDER env var, default to bedrock.

    Returns:
        LLMProvider enum value.

    Raises:
        ValueError: If LLM_PROVIDER has invalid value.
    """
    provider_str = os.getenv("LLM_PROVIDER", "bedrock").lower()
    try:
        return LLMProvider(provider_str)
    except ValueError:
        valid = ", ".join(p.value for p in LLMProvider)
        raise ValueError(
            f"Invalid LLM_PROVIDER '{provider_str}'. Must be one of: {valid}."
        )


def get_agent[T](output_type: type[T], system_prompt: str) -> Agent[None, T]:
    """Create a Pydantic AI agent with Bedrock backend.

    Args:
        output_type: Pydantic model for structured output.
        system_prompt: System prompt for the agent.

    Returns:
        Configured Agent instance.
    """
    from pydantic_ai.models.bedrock import BedrockConverseModel

    from src.config import LLM

    model = BedrockConverseModel(model_name=LLM.model_id)
    return Agent(
        model=model,
        output_type=output_type,
        system_prompt=system_prompt,
    )
