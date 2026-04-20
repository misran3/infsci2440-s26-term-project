"""Naive Bayes topic classifier."""

from __future__ import annotations

from collections import Counter

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from src.loaders.structures import (
    Review,
    Topic,
    TopicClassification,
)


TOPIC_KEYWORDS: dict[str, list[str]] = {
    "performance": ["slow", "fast", "crash", "bug", "freeze", "lag", "speed", "memory", "hang", "error"],
    "usability": ["easy", "difficult", "confusing", "intuitive", "ui", "interface", "simple", "user-friendly"],
    "features": ["feature", "function", "capability", "missing", "wish", "need", "want", "option"],
    "pricing": ["price", "cost", "expensive", "cheap", "value", "subscription", "free", "money", "worth"],
    "support": ["support", "help", "customer service", "documentation", "response", "team"],
    "compatibility": ["install", "compatible", "windows", "mac", "linux", "version", "os"],
}


class TopicClassifier:
    """Multinomial Naive Bayes classifier for review topics."""

    TOPICS: list[str] = [t.value for t in Topic]

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.classifier = MultinomialNB(alpha=1.0)
        self.is_fitted = False

    def fit(self, reviews: list[Review], labels: list[str]) -> dict:
        """
        Train the classifier.

        Args:
            reviews: Training reviews.
            labels: Topic labels (strings matching Topic enum values).

        Returns:
            dict with training metrics.
        """
        if not reviews:
            raise ValueError("Cannot fit with empty reviews list")

        texts = [r.text for r in reviews]
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_fitted = True

        return {
            "n_samples": len(reviews),
            "n_features": len(self.vectorizer.get_feature_names_out()),
            "classes": list(self.classifier.classes_),
        }

    def predict(self, reviews: list[Review]) -> list[TopicClassification]:
        """
        Classify reviews into topics.

        Args:
            reviews: Reviews to classify.

        Returns:
            List of TopicClassification results.

        Raises:
            RuntimeError: If classifier not fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        if not reviews:
            return []

        texts = [r.text for r in reviews]
        X = self.vectorizer.transform(texts)

        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)

        feature_names = self.vectorizer.get_feature_names_out()
        results: list[TopicClassification] = []

        for i, review in enumerate(reviews):
            topic_str = predictions[i]
            try:
                topic = Topic(topic_str)
            except ValueError:
                topic = Topic.OTHER

            confidence = float(probabilities[i].max())

            # Get top contributing features
            topic_idx = list(self.classifier.classes_).index(topic_str)
            feature_scores = X[i].toarray().flatten() * self.classifier.feature_log_prob_[topic_idx]
            top_indices = np.argsort(feature_scores)[-5:][::-1]
            top_features = [
                feature_names[idx]
                for idx in top_indices
                if feature_scores[idx] > 0
            ][:3]

            results.append(
                TopicClassification(
                    review_id=review.review_id,
                    predicted_topic=topic,
                    confidence=confidence,
                    top_features=top_features,
                )
            )

        return results
