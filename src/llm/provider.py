"""Reusable LLM provider using Pydantic AI + Bedrock."""

from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel

from src.config import LLM


def get_agent[T](output_type: type[T], system_prompt: str) -> Agent[None, T]:
    """Create a Pydantic AI agent with Bedrock backend.

    Args:
        output_type: Pydantic model for structured output.
        system_prompt: System prompt for the agent.

    Returns:
        Configured Agent instance.
    """
    model = BedrockConverseModel(model_name=LLM.model_id)
    return Agent(
        model=model,
        output_type=output_type,
        system_prompt=system_prompt,
    )
