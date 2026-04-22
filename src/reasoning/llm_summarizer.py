"""LLM-based summary generation (stub)."""

from __future__ import annotations

from src.loaders.structures import BayesianInsights, Review, SentimentSequence


class LLMSummarizer:
    """Stub for LLM summarization (Person 3 will implement)."""

    def __init__(self) -> None:
        pass

    def summarize(
        self,
        reviews: list[Review],
        insights: BayesianInsights,
        sequences: list[SentimentSequence],
    ) -> str:
        """
        Generate summary (stub returns placeholder).

        Args:
            reviews: Filtered reviews.
            insights: Bayesian insights.
            sequences: Sentiment sequences.

        Returns:
            Placeholder summary string.
        """
        return f"Summary not available - component pending. ({len(reviews)} reviews to analyze)"
