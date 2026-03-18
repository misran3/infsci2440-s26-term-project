"""Data structures for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Topic(str, Enum):
    """Review topic categories."""
    PERFORMANCE = "performance"
    USABILITY = "usability"
    FEATURES = "features"
    PRICING = "pricing"
    SUPPORT = "support"
    COMPATIBILITY = "compatibility"
    OTHER = "other"


class Sentiment(str, Enum):
    """Sentiment states."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class Review:
    """Single review from the dataset."""
    review_id: str
    text: str
    rating: int  # 1-5
    title: str
    product_id: str
    # Added by pipeline stages:
    topic: Topic | None = None
    sentiment: Sentiment | None = None
    sentiment_sequence: list[Sentiment] | None = None
    tfidf_score: float | None = None
    sentences: list[str] | None = None  # Pre-tokenized for HMM
