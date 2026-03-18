"""Mock implementations for testing."""

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
