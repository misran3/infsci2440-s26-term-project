"""Tests for shared data structures."""

from src.loaders.structures import (
    Topic, Sentiment, Review, QueryExpansion,
    TopicClassification, BayesianInsights, SentimentSequence, PipelineResult
)


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


def test_topic_classification_instantiation():
    """TopicClassification can be instantiated with all fields."""
    classification = TopicClassification(
        review_id="R001",
        predicted_topic=Topic.PERFORMANCE,
        confidence=0.85,
        top_features=["crash", "slow", "bug"]
    )
    assert classification.review_id == "R001"
    assert classification.predicted_topic == Topic.PERFORMANCE
    assert classification.confidence == 0.85
    assert classification.top_features == ["crash", "slow", "bug"]


def test_topic_classification_is_frozen():
    """TopicClassification is immutable."""
    classification = TopicClassification(
        review_id="R001",
        predicted_topic=Topic.FEATURES,
        confidence=0.9,
        top_features=[]
    )
    try:
        classification.confidence = 0.5
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass  # Expected


def test_bayesian_insights_instantiation():
    """BayesianInsights can be instantiated with all probability fields."""
    insights = BayesianInsights(
        topic=Topic.PERFORMANCE,
        p_positive_given_topic=0.32,
        p_negative_given_topic=0.68,
        p_high_rating_given_positive=0.85,
        p_low_rating_given_negative=0.75
    )
    assert insights.topic == Topic.PERFORMANCE
    assert insights.p_positive_given_topic == 0.32
    assert insights.p_negative_given_topic == 0.68
    assert insights.p_high_rating_given_positive == 0.85
    assert insights.p_low_rating_given_negative == 0.75


def test_sentiment_sequence_instantiation():
    """SentimentSequence can be instantiated with all fields."""
    sequence = SentimentSequence(
        review_id="R001",
        sentences=["Product arrived fast.", "But packaging was damaged."],
        sentiment_states=[Sentiment.POSITIVE, Sentiment.NEGATIVE],
        transitions={"pos_to_neg": 0.25, "neg_to_pos": 0.6}
    )
    assert sequence.review_id == "R001"
    assert len(sequence.sentences) == 2
    assert sequence.sentiment_states == [Sentiment.POSITIVE, Sentiment.NEGATIVE]
    assert sequence.transitions["pos_to_neg"] == 0.25


def test_pipeline_result_instantiation():
    """PipelineResult aggregates all component outputs."""
    review = Review(
        review_id="R001",
        text="Test review",
        rating=3,
        title="Test",
        product_id="P001"
    )
    expansion = QueryExpansion(
        original_query="test",
        expanded_terms=["test"],
        beam_paths=[]
    )
    classification = TopicClassification(
        review_id="R001",
        predicted_topic=Topic.FEATURES,
        confidence=0.8,
        top_features=["feature"]
    )
    insights = BayesianInsights(
        topic=Topic.FEATURES,
        p_positive_given_topic=0.5,
        p_negative_given_topic=0.5,
        p_high_rating_given_positive=0.8,
        p_low_rating_given_negative=0.8
    )
    sequence = SentimentSequence(
        review_id="R001",
        sentences=["Test review"],
        sentiment_states=[Sentiment.NEUTRAL],
        transitions={}
    )

    result = PipelineResult(
        query="test",
        expansion=expansion,
        candidate_reviews=[review],
        filtered_reviews=[review],
        topic_classifications=[classification],
        bayesian_insights=insights,
        sentiment_sequences=[sequence],
        llm_summary="Test summary"
    )

    assert result.query == "test"
    assert result.expansion.original_query == "test"
    assert len(result.candidate_reviews) == 1
    assert len(result.filtered_reviews) == 1
    assert result.llm_summary == "Test summary"
