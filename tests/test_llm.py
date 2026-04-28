"""Tests for LLM summarizer with fallback behavior."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.loaders.structures import BayesianInsights, Review, SentimentSequence, Topic
from src.reasoning.llm_summarizer import LLMSummarizer


def _insights() -> BayesianInsights:
	return BayesianInsights(
		topic=Topic.PERFORMANCE,
		p_positive_given_topic=0.25,
		p_negative_given_topic=0.75,
		p_high_rating_given_positive=0.8,
		p_low_rating_given_negative=0.7,
	)


@pytest.mark.asyncio
async def test_summarize_no_reviews_returns_no_match_message():
	"""No reviews should produce a clear no-results summary."""
	summarizer = LLMSummarizer()
	text = await summarizer.summarize([], _insights(), [])
	assert "No matching reviews found" in text


@pytest.mark.asyncio
async def test_summarize_fallback_contains_key_stats():
	"""Fallback summary should include count/topic/probability information."""
	summarizer = LLMSummarizer()
	reviews = [
		Review("1", "Crashes often", 1, "Bad", "A"),
		Review("2", "Works perfectly", 5, "Great", "B"),
	]
	sequences = [
		SentimentSequence("1", ["Crashes often"], [], {}),
		SentimentSequence("2", ["Works perfectly"], [], {}),
	]

	text = await summarizer.summarize(reviews, _insights(), sequences)

	assert "Analyzed 2 reviews" in text
	assert "performance" in text
	assert "P(positive|topic)" in text


@pytest.mark.asyncio
async def test_summarize_uses_agent_when_available():
	"""When agent exists and succeeds, summary should come from agent output."""
	summarizer = LLMSummarizer()
	summarizer._agent_init_attempted = True

	mock_result = SimpleNamespace(
		output=SimpleNamespace(
			summary="LLM summary",
			key_themes=["theme1", "theme2"],
			representative_quotes=["quote1"],
		)
	)
	summarizer.agent = SimpleNamespace(run=AsyncMock(return_value=mock_result))

	text = await summarizer.summarize([Review("1", "Text", 4, "T", "P")], _insights(), [])

	assert text == "LLM summary"
	assert summarizer.last_themes == ["theme1", "theme2"]
	assert summarizer.last_quotes == ["quote1"]


@pytest.mark.asyncio
async def test_summarize_agent_failure_uses_error_fallback():
	"""Agent exceptions should produce graceful fallback text."""
	summarizer = LLMSummarizer()
	summarizer._agent_init_attempted = True

	summarizer.agent = SimpleNamespace(run=AsyncMock(side_effect=RuntimeError("API down")))
	text = await summarizer.summarize([Review("1", "Text", 2, "T", "P")], _insights(), [])

	assert "AI-generated summary unavailable" in text


def test_summary_output_has_required_fields():
	"""SummaryOutput model should have summary, key_themes, and representative_quotes."""
	from src.reasoning.llm_summarizer import SummaryOutput

	output = SummaryOutput(
		summary="Test summary",
		key_themes=["theme1", "theme2"],
		representative_quotes=["quote1"],
	)

	assert output.summary == "Test summary"
	assert output.key_themes == ["theme1", "theme2"]
	assert output.representative_quotes == ["quote1"]
