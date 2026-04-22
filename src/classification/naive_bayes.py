"""Naive Bayes topic classifier."""

from __future__ import annotations

from collections import Counter

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from src.loaders.structures import (
    FilterResult,
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

    def save(self, path: str) -> None:
        """
        Save trained classifier to disk.

        Args:
            path: File path to save to.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted classifier")

        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "classifier": self.classifier,
                "is_fitted": self.is_fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "TopicClassifier":
        """
        Load trained classifier from disk.

        Args:
            path: File path to load from.

        Returns:
            Loaded TopicClassifier instance.
        """
        data = joblib.load(path)
        instance = cls()
        instance.vectorizer = data["vectorizer"]
        instance.classifier = data["classifier"]
        instance.is_fitted = data["is_fitted"]
        return instance

    def get_topic_distribution(
        self, classifications: list[TopicClassification]
    ) -> dict[str, int]:
        """
        Get count of reviews per topic.

        Args:
            classifications: Classification results.

        Returns:
            Dict mapping topic name to count.
        """
        counts = Counter(c.predicted_topic.value for c in classifications)
        return dict(counts)

    def detect_topic_from_query(self, query: str) -> str:
        """
        Infer topic from user's query using keyword matching.

        Args:
            query: User's search query.

        Returns:
            Topic string (e.g., "performance", "usability").
        """
        query_lower = query.lower()

        scores: dict[str, int] = {topic: 0 for topic in TOPIC_KEYWORDS}

        for topic, keywords in TOPIC_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    scores[topic] += 1

        best_topic = max(scores, key=lambda k: scores[k])
        return best_topic if scores[best_topic] > 0 else "other"

    def filter_by_topic(
        self,
        reviews: list[Review],
        target_topic: str,
        min_confidence: float = 0.5,
    ) -> FilterResult:
        """
        Filter reviews to those matching target topic.

        Args:
            reviews: Reviews to filter.
            target_topic: Topic to filter for (string).
            min_confidence: Minimum confidence threshold.

        Returns:
            FilterResult with filtered reviews and topic distribution.
        """
        if not reviews:
            return FilterResult(
                filtered_reviews=[],
                classifications=[],
                topic_distribution={},
                fallback_used=False,
            )

        classifications = self.predict(reviews)
        topic_distribution = self.get_topic_distribution(classifications)

        filtered_reviews: list[Review] = []
        filtered_classifications: list[TopicClassification] = []

        try:
            target = Topic(target_topic)
        except ValueError:
            target = Topic.OTHER

        for review, classification in zip(reviews, classifications):
            if (
                classification.predicted_topic == target
                and classification.confidence >= min_confidence
            ):
                filtered_reviews.append(review)
                filtered_classifications.append(classification)

        fallback_used = len(filtered_reviews) == 0 and len(reviews) > 0

        return FilterResult(
            filtered_reviews=filtered_reviews,
            classifications=filtered_classifications,
            topic_distribution=topic_distribution,
            fallback_used=fallback_used,
        )
