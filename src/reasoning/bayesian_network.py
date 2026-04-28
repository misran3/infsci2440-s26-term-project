"""Bayesian Network for probabilistic reasoning over review attributes using pgmpy."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork as PgmpyBN

from src.loaders.structures import BayesianInsights, Review, Topic

logger = logging.getLogger(__name__)


class BayesianNetwork:
    """Probabilistic reasoning over topic, sentiment, and rating using pgmpy."""

    def __init__(self) -> None:
        self.model = PgmpyBN([
            ("topic", "sentiment"),
            ("sentiment", "rating_category"),
        ])
        self._inference: VariableElimination | None = None
        self._is_fitted = False

    @staticmethod
    def _infer_sentiment(review: Review) -> str:
        """Infer sentiment from explicit label when present, else rating."""
        if review.sentiment is not None:
            return review.sentiment.value
        if review.rating >= 4:
            return "positive"
        if review.rating <= 2:
            return "negative"
        return "neutral"

    @staticmethod
    def _rating_category(rating: int) -> str:
        """Categorize rating into low/medium/high."""
        if rating <= 2:
            return "low"
        if rating == 3:
            return "medium"
        return "high"

    @staticmethod
    def _coerce_topic(topic: str) -> Topic:
        """Convert a topic string to Topic enum with safe fallback."""
        try:
            return Topic(topic)
        except ValueError:
            return Topic.OTHER

    def _prepare_dataframe(self, reviews: list[Review]) -> pd.DataFrame:
        """Convert reviews to DataFrame for pgmpy."""
        data = []
        for r in reviews:
            topic_val = r.topic.value if r.topic else "other"
            data.append({
                "topic": topic_val,
                "sentiment": self._infer_sentiment(r),
                "rating_category": self._rating_category(r.rating),
            })
        return pd.DataFrame(data)

    def fit(self, reviews: list[Review]) -> None:
        """Fit CPTs using Maximum Likelihood Estimation."""
        if not reviews:
            return

        df = self._prepare_dataframe(reviews)
        self.model.fit(df, estimator=MaximumLikelihoodEstimator)
        self._inference = VariableElimination(self.model)
        self._is_fitted = True

    def infer(self, reviews: list[Review], topic: str = "other") -> BayesianInsights:
        """Query the network for probabilistic insights about a topic.

        Args:
            reviews: Unused. Kept for interface compatibility with pipeline.
                     Inference uses the fitted model, not runtime reviews.
            topic: Topic string to query probabilities for.

        Returns:
            BayesianInsights with conditional probabilities.
        """
        del reviews  # Explicitly unused; inference uses fitted model
        t = self._coerce_topic(topic)
        topic_str = t.value

        if not self._is_fitted:
            return BayesianInsights(
                topic=t,
                p_positive_given_topic=0.5,
                p_negative_given_topic=0.5,
                p_high_rating_given_positive=0.5,
                p_low_rating_given_negative=0.5,
            )

        try:
            sentiment_dist = self._inference.query(
                variables=["sentiment"],
                evidence={"topic": topic_str},
            )
            sentiment_values = sentiment_dist.values
            sentiment_states = sentiment_dist.state_names["sentiment"]

            p_pos = float(sentiment_values[sentiment_states.index("positive")]) if "positive" in sentiment_states else 0.0
            p_neg = float(sentiment_values[sentiment_states.index("negative")]) if "negative" in sentiment_states else 0.0
        except (KeyError, ValueError, IndexError) as e:
            logger.warning("Failed to query sentiment distribution for topic '%s': %s", topic_str, e)
            p_pos, p_neg = 0.5, 0.5

        try:
            rating_given_pos = self._inference.query(
                variables=["rating_category"],
                evidence={"sentiment": "positive"},
            )
            rating_states = rating_given_pos.state_names["rating_category"]
            p_high_given_pos = float(rating_given_pos.values[rating_states.index("high")]) if "high" in rating_states else 0.5
        except (KeyError, ValueError, IndexError) as e:
            logger.warning("Failed to query rating distribution for positive sentiment: %s", e)
            p_high_given_pos = 0.5

        try:
            rating_given_neg = self._inference.query(
                variables=["rating_category"],
                evidence={"sentiment": "negative"},
            )
            rating_states = rating_given_neg.state_names["rating_category"]
            p_low_given_neg = float(rating_given_neg.values[rating_states.index("low")]) if "low" in rating_states else 0.5
        except (KeyError, ValueError, IndexError) as e:
            logger.warning("Failed to query rating distribution for negative sentiment: %s", e)
            p_low_given_neg = 0.5

        return BayesianInsights(
            topic=t,
            p_positive_given_topic=p_pos,
            p_negative_given_topic=p_neg,
            p_high_rating_given_positive=p_high_given_pos,
            p_low_rating_given_negative=p_low_given_neg,
        )

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        """Save the fitted model to disk.

        Args:
            path: File path to save the model to.
            metadata: Optional training metadata to embed.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted BayesianNetwork")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "BayesianNetwork":
        """Load a fitted BayesianNetwork from disk.

        Args:
            path: File path to load the model from.

        Returns:
            A fitted BayesianNetwork instance.

        Raises:
            ValueError: If the loaded object is not a valid pgmpy DiscreteBayesianNetwork.
        """
        data = joblib.load(path)

        # Handle old format (raw model) vs new format (dict with model + metadata)
        if isinstance(data, PgmpyBN):
            # Old format: raw pgmpy model
            model = data
            metadata = {}
        elif isinstance(data, dict) and "model" in data:
            # New format: dict with model and metadata
            model = data["model"]
            metadata = data.get("metadata", {})
        else:
            raise ValueError(
                f"Invalid model type: expected DiscreteBayesianNetwork or dict, got {type(data).__name__}"
            )

        if not isinstance(model, PgmpyBN):
            raise ValueError(
                f"Invalid model type: expected DiscreteBayesianNetwork, got {type(model).__name__}"
            )

        # Verify model has fitted CPDs
        cpds = model.get_cpds()
        if not cpds or not all(isinstance(cpd, TabularCPD) for cpd in cpds):
            raise ValueError("Loaded model does not have valid fitted CPDs")

        instance = cls()
        instance.model = model
        instance._inference = VariableElimination(instance.model)
        instance._is_fitted = True
        instance.metadata = metadata
        return instance
