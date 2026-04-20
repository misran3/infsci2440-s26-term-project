"""LLM-based term filtering for search expansion."""

from pydantic import BaseModel


class FilteredTerms(BaseModel):
    """LLM response model for filtered terms."""

    relevant_terms: list[str]
