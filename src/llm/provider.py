"""Reusable LLM provider using Pydantic AI + Bedrock."""

from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel

from src.config import LLM


def get_agent[T](result_type: type[T], system_prompt: str) -> Agent[None, T]:
    """Create a Pydantic AI agent with Bedrock backend.

    Args:
        result_type: Pydantic model for structured output.
        system_prompt: System prompt for the agent.

    Returns:
        Configured Agent instance.
    """
    model = BedrockConverseModel(
        model_id=LLM.model_id,
        region_name=LLM.region,
    )
    return Agent(
        model=model,
        result_type=result_type,
        system_prompt=system_prompt,
    )
