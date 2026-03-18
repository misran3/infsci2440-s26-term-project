"""Tests for mock implementations."""

from src.mocks import MOCK_REVIEWS
from src.loaders.structures import Review


def test_mock_reviews_exist():
    """MOCK_REVIEWS contains sample reviews."""
    assert len(MOCK_REVIEWS) >= 3


def test_mock_reviews_are_review_instances():
    """MOCK_REVIEWS contains Review instances."""
    for review in MOCK_REVIEWS:
        assert isinstance(review, Review)


def test_mock_reviews_have_varied_ratings():
    """MOCK_REVIEWS covers different rating values."""
    ratings = {r.rating for r in MOCK_REVIEWS}
    assert 1 in ratings  # Negative
    assert 5 in ratings  # Positive


def test_mock_reviews_have_sentences():
    """MOCK_REVIEWS have pre-tokenized sentences."""
    for review in MOCK_REVIEWS:
        assert review.sentences is not None
        assert len(review.sentences) >= 1


from src.mocks import mock_beam_search
from src.loaders.structures import QueryExpansion


def test_mock_beam_search_returns_query_expansion():
    """mock_beam_search returns QueryExpansion instance."""
    result = mock_beam_search("shipping")
    assert isinstance(result, QueryExpansion)


def test_mock_beam_search_includes_original_query():
    """mock_beam_search preserves original query."""
    result = mock_beam_search("shipping")
    assert result.original_query == "shipping"
    assert "shipping" in result.expanded_terms


def test_mock_beam_search_expands_terms():
    """mock_beam_search adds expanded terms."""
    result = mock_beam_search("test")
    assert len(result.expanded_terms) > 1


def test_mock_beam_search_has_beam_paths():
    """mock_beam_search includes beam paths with scores."""
    result = mock_beam_search("query")
    assert len(result.beam_paths) > 0
    assert "path" in result.beam_paths[0]
    assert "score" in result.beam_paths[0]
