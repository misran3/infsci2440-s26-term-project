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


@dataclass(frozen=True)
class BayesianInsights:
    """Probabilistic insights from Bayesian Network."""
    topic: Topic
    p_positive_given_topic: float
    p_negative_given_topic: float
    p_high_rating_given_positive: float  # P(4-5 stars | positive)
    p_low_rating_given_negative: float   # P(1-2 stars | negative)


@dataclass(frozen=True)
class SentimentSequence:
    """HMM analysis of sentiment flow in a review."""
    review_id: str
    sentences: list[str]
    sentiment_states: list[Sentiment]
    transitions: dict[str, float]  # {"pos_to_neg": 0.25, ...}


@dataclass(frozen=True)
class PipelineResult:
    """Final result passed to LLM and UI."""
    query: str
    expansion: QueryExpansion
    filtered_terms: list[str]
    candidate_reviews: list[Review]
    filtered_reviews: list[Review]
    topic_classifications: list[TopicClassification]
    bayesian_insights: BayesianInsights
    sentiment_sequences: list[SentimentSequence]
    llm_summary: str
    llm_themes: list[str] | None = None
    llm_quotes: list[str] | None = None
    preprocessed: PreprocessedQuery | None = None


@dataclass(frozen=True)
class FilterResult:
    """Result of topic filtering."""
    filtered_reviews: list[Review]
    classifications: list[TopicClassification]
    topic_distribution: dict[str, int]
    fallback_used: bool


@dataclass(frozen=True)
class PreprocessedQuery:
    """Result of query preprocessing."""
    original_query: str
    extracted_keywords: list[str]
    was_preprocessed: bool
