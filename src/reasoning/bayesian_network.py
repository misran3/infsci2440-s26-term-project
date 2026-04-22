"""Bayesian network for review insights (stub)."""

from __future__ import annotations

from src.loaders.structures import BayesianInsights, Review, Topic


class BayesianNetwork:
    """Stub for Bayesian Network reasoning (Person 3 will implement)."""

    def __init__(self) -> None:
        pass

    def fit(self, reviews: list[Review]) -> None:
        """Fit the network (stub - does nothing)."""
        pass

    def infer(self, reviews: list[Review], topic: str = "other") -> BayesianInsights:
        """
        Return probabilistic insights (stub returns defaults).

        Args:
            reviews: Reviews to analyze.
            topic: Topic for conditional probabilities.

        Returns:
            BayesianInsights with default values.
        """
        try:
            t = Topic(topic)
        except ValueError:
            t = Topic.OTHER

        return BayesianInsights(
            topic=t,
            p_positive_given_topic=0.0,
            p_negative_given_topic=0.0,
            p_high_rating_given_positive=0.0,
            p_low_rating_given_negative=0.0,
        )
