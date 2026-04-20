"""HMM for sentiment sequence analysis (stub)."""

from __future__ import annotations

from src.loaders.structures import Review, SentimentSequence


class HMMSentiment:
    """Stub for HMM sentiment analysis (Person 3 will implement)."""

    def __init__(self) -> None:
        pass

    def fit(self, reviews: list[Review]) -> None:
        """Fit the HMM (stub - does nothing)."""
        pass

    def analyze(self, reviews: list[Review]) -> list[SentimentSequence]:
        """
        Analyze sentiment sequences (stub returns empty).

        Args:
            reviews: Reviews to analyze.

        Returns:
            List of SentimentSequence (empty for stub).
        """
        return [
            SentimentSequence(
                review_id=r.review_id,
                sentences=r.sentences or [],
                sentiment_states=[],
                transitions={},
            )
            for r in reviews
        ]
