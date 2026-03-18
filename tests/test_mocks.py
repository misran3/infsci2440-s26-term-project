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
