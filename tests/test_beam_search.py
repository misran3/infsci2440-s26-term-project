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


def test_empty_query():
    """Empty query should return empty expansion."""
    expander = BeamSearchExpander()
    result = expander.expand("")

    assert result.original_query == ""
    assert result.expanded_terms == []
    assert result.beam_paths == []


def test_single_word_query():
    """Single word query should work."""
    expander = BeamSearchExpander()
    result = expander.expand("fast")

    assert result.original_query == "fast"
    assert "fast" in result.expanded_terms


def test_unknown_word_no_synonyms():
    """Unknown word should fall back to original term."""
    expander = BeamSearchExpander()
    result = expander.expand("xyzabc123")

    assert result.original_query == "xyzabc123"
    assert "xyzabc123" in result.expanded_terms


def test_beam_width_limits_candidates():
    """Smaller beam width should produce fewer paths."""
    expander_narrow = BeamSearchExpander(beam_width=1, max_depth=2)
    expander_wide = BeamSearchExpander(beam_width=3, max_depth=2)

    result_narrow = expander_narrow.expand("good product")
    result_wide = expander_wide.expand("good product")

    assert len(result_narrow.beam_paths) <= len(result_wide.beam_paths)


def test_depth_zero_returns_original():
    """Depth 0 should return only original terms."""
    expander = BeamSearchExpander(max_depth=0)
    result = expander.expand("fast shipping")

    assert result.original_query == "fast shipping"
    assert set(result.expanded_terms) == {"fast", "shipping"}
    assert result.beam_paths == []


def test_synonyms_exclude_underscores():
    """Synonyms with underscores should be filtered out."""
    expander = BeamSearchExpander()
    synonyms = expander.get_synonyms("new")

    for syn in synonyms:
        assert "_" not in syn
