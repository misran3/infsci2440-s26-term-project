"""Data structures for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


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


@dataclass(frozen=True)
class QueryExpansion:
    """Result of beam search query expansion."""
    original_query: str
    expanded_terms: list[str]
    beam_paths: list[dict[str, Any]]  # [{"path": [...], "score": 0.9}, ...]


@dataclass(frozen=True)
class TopicClassification:
    """Result of Naive Bayes classification."""
    review_id: str
    predicted_topic: Topic
    confidence: float
    top_features: list[str]
