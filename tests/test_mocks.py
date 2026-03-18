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


from src.mocks import mock_tfidf_retrieve


def test_mock_tfidf_retrieve_returns_reviews():
    """mock_tfidf_retrieve returns list of Review instances."""
    result = mock_tfidf_retrieve(["test"], top_k=10)
    assert isinstance(result, list)
    assert all(isinstance(r, Review) for r in result)


def test_mock_tfidf_retrieve_respects_top_k():
    """mock_tfidf_retrieve limits results to top_k."""
    result = mock_tfidf_retrieve(["test"], top_k=2)
    assert len(result) <= 2


def test_mock_tfidf_retrieve_sets_tfidf_score():
    """mock_tfidf_retrieve assigns tfidf_score to reviews."""
    result = mock_tfidf_retrieve(["test"], top_k=10)
    for review in result:
        assert review.tfidf_score is not None
        assert review.tfidf_score >= 0


from src.mocks import mock_classify
from src.loaders.structures import TopicClassification, Topic


def test_mock_classify_returns_classifications():
    """mock_classify returns TopicClassification for each review."""
    reviews = MOCK_REVIEWS[:2]
    result = mock_classify(reviews)
    assert len(result) == len(reviews)
    assert all(isinstance(c, TopicClassification) for c in result)


def test_mock_classify_matches_review_ids():
    """mock_classify uses correct review_ids."""
    reviews = MOCK_REVIEWS[:2]
    result = mock_classify(reviews)
    for i, classification in enumerate(result):
        assert classification.review_id == reviews[i].review_id


def test_mock_classify_has_valid_confidence():
    """mock_classify returns valid confidence scores."""
    result = mock_classify(MOCK_REVIEWS)
    for classification in result:
        assert 0 <= classification.confidence <= 1
