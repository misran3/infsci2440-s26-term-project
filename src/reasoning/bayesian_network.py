"""Bayesian-style probabilistic reasoning over review attributes."""

from __future__ import annotations

from src.loaders.structures import BayesianInsights, Review, Sentiment, Topic


class BayesianNetwork:
    """Probabilistic reasoning over topic, sentiment, and rating."""

    def __init__(self) -> None:
        self._fitted_reviews: list[Review] = []

    def fit(self, reviews: list[Review]) -> None:
        """Store fitted reviews for later inference."""
        self._fitted_reviews = list(reviews)

    @staticmethod
    def _coerce_topic(topic: str) -> Topic:
        """Convert a topic string to Topic enum with safe fallback."""
        try:
            return Topic(topic)
        except ValueError:
            return Topic.OTHER

    @staticmethod
    def _infer_sentiment(review: Review) -> Sentiment:
        """Infer sentiment from explicit label when present, else rating."""
        if review.sentiment is not None:
            return review.sentiment

        if review.rating >= 4:
            return Sentiment.POSITIVE
        if review.rating <= 2:
            return Sentiment.NEGATIVE
        return Sentiment.NEUTRAL

    @staticmethod
    def _smoothed_ratio(numerator: int, denominator: int, alpha: float = 1.0) -> float:
        """Laplace-smoothed Bernoulli ratio."""
        return float((numerator + alpha) / (denominator + (2 * alpha)))

    def infer(self, reviews: list[Review], topic: str = "other") -> BayesianInsights:
        """
        Return probabilistic insights (stub returns defaults).

        Args:
            reviews: Reviews to analyze.
            topic: Topic for conditional probabilities.

        Returns:
            BayesianInsights with default values.
        """
        t = self._coerce_topic(topic)

        # Prefer the reviews provided for this query; fall back to fitted reviews.
        base_reviews = list(reviews) if reviews else self._fitted_reviews

        if not base_reviews:
            return BayesianInsights(
                topic=t,
                p_positive_given_topic=0.5,
                p_negative_given_topic=0.5,
                p_high_rating_given_positive=0.5,
                p_low_rating_given_negative=0.5,
            )

        topic_reviews = [r for r in base_reviews if (r.topic or Topic.OTHER) == t]
        if not topic_reviews:
            topic_reviews = base_reviews

        topic_sentiments = [self._infer_sentiment(r) for r in topic_reviews]
        pos_topic = sum(1 for s in topic_sentiments if s == Sentiment.POSITIVE)
        neg_topic = sum(1 for s in topic_sentiments if s == Sentiment.NEGATIVE)
        total_topic = len(topic_sentiments)

        all_pos = [r for r in base_reviews if self._infer_sentiment(r) == Sentiment.POSITIVE]
        all_neg = [r for r in base_reviews if self._infer_sentiment(r) == Sentiment.NEGATIVE]

        high_given_pos_num = sum(1 for r in all_pos if r.rating >= 4)
        low_given_neg_num = sum(1 for r in all_neg if r.rating <= 2)

        p_positive_given_topic = self._smoothed_ratio(pos_topic, total_topic)
        p_negative_given_topic = self._smoothed_ratio(neg_topic, total_topic)
        p_high_rating_given_positive = self._smoothed_ratio(high_given_pos_num, len(all_pos))
        p_low_rating_given_negative = self._smoothed_ratio(low_given_neg_num, len(all_neg))

        return BayesianInsights(
            topic=t,
            p_positive_given_topic=p_positive_given_topic,
            p_negative_given_topic=p_negative_given_topic,
            p_high_rating_given_positive=p_high_rating_given_positive,
            p_low_rating_given_negative=p_low_rating_given_negative,
        )
