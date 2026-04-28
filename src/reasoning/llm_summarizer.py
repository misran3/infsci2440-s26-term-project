"""LLM-based summary generation with structured output."""

from __future__ import annotations

import os

from pydantic import BaseModel

from src.llm import get_agent
from src.loaders.structures import BayesianInsights, Review, SentimentSequence


SUMMARY_PROMPT = """You are an AI assistant that summarizes product review analysis results.
Given pipeline analysis data and sample reviews, write a concise summary that:
1. States the key finding (topic, number of reviews, sentiment distribution)
2. Highlights 2-3 main themes or patterns
3. Includes 2-3 representative quotes from the actual reviews

Be concise and actionable. Do not invent quotes - only use text from the provided reviews.
Keep the summary under 150 words."""


class SummaryOutput(BaseModel):
    """Structured output for LLM summary responses."""

    summary: str
    key_themes: list[str]
    representative_quotes: list[str]


class LLMSummarizer:
    """Summarize pipeline results with optional LLM generation."""

    def __init__(self) -> None:
        self.agent = None
        self._agent_init_attempted = False
        self.last_themes: list[str] = []
        self.last_quotes: list[str] = []

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
    def _format_reviews(reviews: list[Review], max_reviews: int = 10) -> str:
        """Format reviews for LLM context."""
        lines = []
        for r in reviews[:max_reviews]:
            stars = "*" * r.rating
            text = r.text[:300] + "..." if len(r.text) > 300 else r.text
            lines.append(f"[{stars}] {text}")
        return "\n\n".join(lines)

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

    async def summarize(
        self,
        reviews: list[Review],
        insights: BayesianInsights,
        sequences: list[SentimentSequence],
    ) -> str:
        """Generate a natural-language summary."""
        self.last_themes = []
        self.last_quotes = []

        fallback = self._fallback_summary(reviews, insights, sequences)

        self._maybe_init_agent()

        if self.agent is None:
            return fallback

        reviews_text = self._format_reviews(reviews)

        sequence_preview = []
        for seq in sequences[:5]:
            states = [s.value for s in seq.sentiment_states]
            sequence_preview.append({
                "review_id": seq.review_id,
                "states": states,
            })

        prompt = (
            f"Review count: {len(reviews)}\n"
            f"Topic: {insights.topic.value}\n"
            f"P(positive|topic): {insights.p_positive_given_topic:.4f}\n"
            f"P(negative|topic): {insights.p_negative_given_topic:.4f}\n"
            f"P(high_rating|positive): {insights.p_high_rating_given_positive:.4f}\n"
            f"P(low_rating|negative): {insights.p_low_rating_given_negative:.4f}\n"
            f"Sample sentiment sequences: {sequence_preview}\n\n"
            f"SAMPLE REVIEWS:\n{reviews_text}"
        )

        try:
            result = await self.agent.run(prompt)
            output = result.output
            self.last_themes = output.key_themes
            self.last_quotes = output.representative_quotes
            summary = output.summary.strip()
            return summary if summary else fallback
        except Exception:
            return f"AI-generated summary unavailable. {fallback}"
