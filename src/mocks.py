"""Mock implementations for testing."""

import copy

from src.loaders.structures import (
    Topic, Sentiment, Review, QueryExpansion,
    TopicClassification, BayesianInsights, SentimentSequence, PipelineResult
)


MOCK_REVIEWS = [
    Review(
        review_id="MOCK001",
        text="Software crashed on startup. Very frustrating experience.",
        rating=1,
        title="Crashes constantly",
        product_id="P001",
        sentences=["Software crashed on startup.", "Very frustrating experience."]
    ),
    Review(
        review_id="MOCK002",
        text="Easy to use interface. Love the new features!",
        rating=5,
        title="Great software",
        product_id="P002",
        sentences=["Easy to use interface.", "Love the new features!"]
    ),
    Review(
        review_id="MOCK003",
        text="Price is too high for what you get. Support was helpful though.",
        rating=3,
        title="Mixed feelings",
        product_id="P003",
        sentences=["Price is too high for what you get.", "Support was helpful though."]
    ),
]


def mock_beam_search(query: str) -> QueryExpansion:
    """Mock beam search expansion."""
    return QueryExpansion(
        original_query=query,
        expanded_terms=[query, f"{query} issues", f"{query} problems"],
        beam_paths=[
            {"path": [query, f"{query} issues"], "score": 0.9},
            {"path": [query, f"{query} problems"], "score": 0.85},
        ]
    )


def mock_tfidf_retrieve(terms: list[str], top_k: int = 10) -> list[Review]:
    """Mock TF-IDF retrieval."""
    reviews = [copy.copy(r) for r in MOCK_REVIEWS]
    for i, r in enumerate(reviews):
        r.tfidf_score = 1.0 - (i * 0.1)  # Descending scores
    return reviews[:top_k]


def mock_classify(reviews: list[Review]) -> list[TopicClassification]:
    """Mock topic classification."""
    return [
        TopicClassification(
            review_id=r.review_id,
            predicted_topic=Topic.PERFORMANCE,
            confidence=0.85,
            top_features=["crash", "slow", "bug"]
        )
        for r in reviews
    ]


def mock_bayesian_query(topic: Topic) -> BayesianInsights:
    """Mock Bayesian network query."""
    return BayesianInsights(
        topic=topic,
        p_positive_given_topic=0.32,
        p_negative_given_topic=0.68,
        p_high_rating_given_positive=0.85,
        p_low_rating_given_negative=0.75,
    )
