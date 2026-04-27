"""LLM-based summary generation with deterministic fallback."""

from __future__ import annotations

import os

from pydantic import BaseModel

from src.llm import get_agent
from src.loaders.structures import BayesianInsights, Review, SentimentSequence


SUMMARY_PROMPT = """You summarize findings from software review analysis.
Use concise, factual language and mention:
1) number of relevant reviews
2) positive vs negative topic probabilities
3) one short sentiment-flow observation
Do not invent facts. Keep the summary under 120 words."""


class SummaryOutput(BaseModel):
    """Structured output for LLM summary responses."""

    summary: str


class LLMSummarizer:
    """Summarize pipeline results with optional LLM generation."""

    def __init__(self) -> None:
        self.agent = None
        self._agent_init_attempted = False

    def _maybe_init_agent(self) -> None:
        """Initialize agent lazily and only when explicitly enabled."""
        if self._agent_init_attempted:
            return
        self._agent_init_attempted = True

        enabled = os.getenv("LLM_SUMMARY_ENABLED", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        if not enabled:
            return

        self.agent = get_agent(
            output_type=SummaryOutput,
            system_prompt=SUMMARY_PROMPT,
            required=False,
        )

    @staticmethod
    def _fallback_summary(
        reviews: list[Review], insights: BayesianInsights, sequences: list[SentimentSequence]
    ) -> str:
        """Generate robust fallback when LLM is unavailable."""
        if not reviews:
            return "No matching reviews found for the current filters and query."

        avg_rating = sum(r.rating for r in reviews) / len(reviews)
        avg_sentences = (
            sum(len(seq.sentiment_states) for seq in sequences) / len(sequences)
            if sequences
            else 0.0
        )

        return (
            f"Analyzed {len(reviews)} reviews for topic '{insights.topic.value}'. "
            f"P(positive|topic)={insights.p_positive_given_topic:.2f}, "
            f"P(negative|topic)={insights.p_negative_given_topic:.2f}. "
            f"Average rating is {avg_rating:.2f}/5. "
            f"Average sentiment-sequence length is {avg_sentences:.1f} sentences."
        )

    def summarize(
        self,
        reviews: list[Review],
        insights: BayesianInsights,
        sequences: list[SentimentSequence],
    ) -> str:
        """
        Generate a natural-language summary.

        Args:
            reviews: Filtered reviews.
            insights: Bayesian insights.
            sequences: Sentiment sequences.

        Returns:
            Summary string.
        """
        fallback = self._fallback_summary(reviews, insights, sequences)

        self._maybe_init_agent()

        if self.agent is None:
            return fallback

        sequence_preview = []
        for seq in sequences[:5]:
            states = [s.value for s in seq.sentiment_states]
            sequence_preview.append(
                {
                    "review_id": seq.review_id,
                    "states": states,
                    "transitions": seq.transitions,
                }
            )

        prompt = (
            f"Review count: {len(reviews)}\n"
            f"Topic: {insights.topic.value}\n"
            f"P(positive|topic): {insights.p_positive_given_topic:.4f}\n"
            f"P(negative|topic): {insights.p_negative_given_topic:.4f}\n"
            f"P(high_rating|positive): {insights.p_high_rating_given_positive:.4f}\n"
            f"P(low_rating|negative): {insights.p_low_rating_given_negative:.4f}\n"
            f"Sample sentiment sequences: {sequence_preview}"
        )

        try:
            result = self.agent.run_sync(prompt)
            summary = result.output.summary.strip()
            return summary if summary else fallback
        except Exception:
            return f"AI-generated summary unavailable. {fallback}"
