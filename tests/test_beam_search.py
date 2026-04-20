"""Tests for beam search expansion."""

import pytest
from src.search.beam_search import BeamSearchExpander


def test_expand_returns_synonyms():
    """Basic expansion should return original terms plus synonyms."""
    expander = BeamSearchExpander()
    result = expander.expand("fast shipping")

    assert result.original_query == "fast shipping"
    assert "fast" in result.expanded_terms
    assert "shipping" in result.expanded_terms
    assert len(result.expanded_terms) > 2  # Should have synonyms
