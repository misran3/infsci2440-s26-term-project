"""Tests for shared data structures."""

from src.loaders.structures import Topic, Sentiment, Review, QueryExpansion


def test_topic_enum_has_expected_values():
    """Topic enum contains all expected categories."""
    expected = {"performance", "usability", "features", "pricing", "support", "compatibility", "other"}
    actual = {t.value for t in Topic}
    assert actual == expected


def test_topic_enum_string_comparison():
    """Topic enum supports direct string comparison via str base."""
    assert Topic.PERFORMANCE == "performance"
    assert Topic.USABILITY == "usability"


def test_sentiment_enum_has_expected_values():
    """Sentiment enum contains positive, negative, neutral."""
    expected = {"positive", "negative", "neutral"}
    actual = {s.value for s in Sentiment}
    assert actual == expected


def test_sentiment_enum_string_comparison():
    """Sentiment enum supports direct string comparison."""
    assert Sentiment.POSITIVE == "positive"
    assert Sentiment.NEGATIVE == "negative"
    assert Sentiment.NEUTRAL == "neutral"


def test_review_required_fields():
    """Review can be instantiated with required fields only."""
    review = Review(
        review_id="R001",
        text="Great product!",
        rating=5,
        title="Love it",
        product_id="P001"
    )
    assert review.review_id == "R001"
    assert review.text == "Great product!"
    assert review.rating == 5
    assert review.title == "Love it"
    assert review.product_id == "P001"


def test_review_optional_fields_default_to_none():
    """Review optional fields default to None."""
    review = Review(
        review_id="R001",
        text="Great product!",
        rating=5,
        title="Love it",
        product_id="P001"
    )
    assert review.topic is None
    assert review.sentiment is None
    assert review.sentiment_sequence is None
    assert review.tfidf_score is None
    assert review.sentences is None


def test_review_is_mutable():
    """Review fields can be modified (not frozen)."""
    review = Review(
        review_id="R001",
        text="Great product!",
        rating=5,
        title="Love it",
        product_id="P001"
    )
    review.topic = Topic.FEATURES
    review.sentiment = Sentiment.POSITIVE
    assert review.topic == Topic.FEATURES
    assert review.sentiment == Sentiment.POSITIVE


def test_query_expansion_instantiation():
    """QueryExpansion can be instantiated with all fields."""
    expansion = QueryExpansion(
        original_query="shipping",
        expanded_terms=["shipping", "delivery", "arrived"],
        beam_paths=[{"path": ["shipping", "delivery"], "score": 0.9}]
    )
    assert expansion.original_query == "shipping"
    assert expansion.expanded_terms == ["shipping", "delivery", "arrived"]
    assert expansion.beam_paths[0]["score"] == 0.9


def test_query_expansion_is_frozen():
    """QueryExpansion is immutable."""
    expansion = QueryExpansion(
        original_query="test",
        expanded_terms=["test"],
        beam_paths=[]
    )
    try:
        expansion.original_query = "modified"
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass  # Expected - dataclass is frozen
