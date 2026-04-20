"""Tests for beam search expansion."""

from src.search.beam_search import BeamSearchExpander


def test_expand_returns_synonyms():
    """Basic expansion should return original terms plus synonyms."""
    expander = BeamSearchExpander()
    result = expander.expand("fast shipping")

    assert result.original_query == "fast shipping"
    assert "fast" in result.expanded_terms
    assert "shipping" in result.expanded_terms
    assert len(result.expanded_terms) > 2  # Should have synonyms


def test_beam_paths_tracked():
    """Beam paths should be tracked for visualization."""
    expander = BeamSearchExpander(beam_width=2, max_depth=1)
    result = expander.expand("good product")

    assert len(result.beam_paths) > 0
    assert "path" in result.beam_paths[0]
    assert "score" in result.beam_paths[0]
    assert isinstance(result.beam_paths[0]["path"], list)
    assert isinstance(result.beam_paths[0]["score"], float)
