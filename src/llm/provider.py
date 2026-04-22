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


def _check_bedrock_credentials() -> bool:
    """Check if AWS credentials can access Bedrock.

    Returns:
        True if credentials are valid, False otherwise.
    """
    try:
        import boto3

        client = boto3.client("bedrock")
        client.list_foundation_models(maxResults=1)
        return True
    except Exception:
        return False


def validate_credentials(provider: LLMProvider) -> tuple[bool, str]:
    """Check if credentials exist for provider.

    Args:
        provider: The LLM provider to validate.

    Returns:
        Tuple of (valid, error_message). If valid, error_message is empty.
    """
    if provider == LLMProvider.OPENAI:
        if not os.getenv("OPENAI_API_KEY"):
            return (
                False,
                "LLM provider is set to 'openai' but OPENAI_API_KEY environment "
                "variable is not set.",
            )
        return True, ""

    if provider == LLMProvider.BEDROCK:
        if not _check_bedrock_credentials():
            return (
                False,
                "LLM provider is set to 'bedrock' but AWS credentials are not "
                "configured. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, "
                "or configure AWS CLI.",
            )
        return True, ""

    return False, f"Unknown provider: {provider}"


# Default models per provider
_DEFAULT_MODELS = {
    LLMProvider.BEDROCK: "us.anthropic.claude-sonnet-4-6",
    LLMProvider.OPENAI: "gpt-4o-mini",
}


def get_model():
    """Create appropriate Pydantic AI model based on provider and model env vars.

    Returns:
        Configured model instance (BedrockConverseModel or OpenAIModel).
    """
    provider = get_provider()

    if provider == LLMProvider.OPENAI:
        from pydantic_ai.models.openai import OpenAIModel

        model_name = os.getenv("OPENAI_MODEL", _DEFAULT_MODELS[LLMProvider.OPENAI])
        return OpenAIModel(model_name)

    from pydantic_ai.models.bedrock import BedrockConverseModel

    model_name = os.getenv("BEDROCK_MODEL", _DEFAULT_MODELS[LLMProvider.BEDROCK])
    return BedrockConverseModel(model_name=model_name)


def get_agent[T](
    output_type: type[T],
    system_prompt: str,
    *,
    required: bool = True,
) -> Agent[None, T] | None:
    """Create a Pydantic AI agent with multi-backend support.

    Args:
        output_type: Pydantic model for structured output.
        system_prompt: System prompt for the agent.
        required: If True, raise LLMNotAvailableError when credentials invalid.
                  If False, return None when credentials invalid.

    Returns:
        Configured Agent instance, or None if required=False and credentials invalid.

    Raises:
        LLMNotAvailableError: If required=True and credentials are invalid.
    """
    provider = get_provider()
    valid, error_msg = validate_credentials(provider)

    if not valid:
        if required:
            raise LLMNotAvailableError(error_msg)
        return None

    model = get_model()
    return Agent(
        model=model,
        output_type=output_type,
        system_prompt=system_prompt,
    )
