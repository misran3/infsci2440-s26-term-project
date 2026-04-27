"""Integration tests for full pipeline."""

import asyncio

from src.classification.naive_bayes import TopicClassifier
from src.loaders.structures import Review
from src.pipeline import PipelineComponents, SurveyAnalysisPipeline
from src.reasoning.bayesian_network import BayesianNetwork
from src.reasoning.hmm_sentiment import HMMSentiment
from src.reasoning.llm_summarizer import LLMSummarizer
from src.search.beam_search import BeamSearchExpander
from src.search.tfidf_retriever import TFIDFRetriever


def _build_pipeline() -> SurveyAnalysisPipeline:
	corpus = [
		Review("1", "App crashes frequently and feels slow", 1, "Bad", "A"),
		Review("2", "Very easy interface and simple setup", 5, "Great", "B"),
		Review("3", "Subscription cost is too high", 2, "Expensive", "C"),
		Review("4", "Frequent bugs after update", 1, "Buggy", "D"),
	]

	expander = BeamSearchExpander(beam_width=3, max_depth=1)
	retriever = TFIDFRetriever(corpus)
	retriever.fit()

	classifier = TopicClassifier()
	classifier.fit(
		corpus,
		["performance", "usability", "pricing", "performance"],
	)

	components = PipelineComponents(
		expander=expander,
		retriever=retriever,
		classifier=classifier,
		bayesian_net=BayesianNetwork(),
		hmm=HMMSentiment(),
		summarizer=LLMSummarizer(),
		term_filter=None,
	)
	return SurveyAnalysisPipeline(components)


def test_pipeline_happy_path_returns_all_sections():
	"""Pipeline should return complete structured output for normal query."""
	pipeline = _build_pipeline()

	result, filter_result = asyncio.run(
		pipeline.run("app crashes", topic_filter="performance", min_confidence=0.2, top_k=10)
	)

	assert result.query == "app crashes"
	assert result.expansion is not None
	assert len(result.candidate_reviews) > 0
	assert result.bayesian_insights is not None
	assert isinstance(result.llm_summary, str)
	assert len(result.sentiment_sequences) == len(result.filtered_reviews or result.candidate_reviews)
	assert isinstance(filter_result.fallback_used, bool)


def test_pipeline_empty_query_handles_gracefully():
	"""Empty query should not crash and should return safe defaults."""
	pipeline = _build_pipeline()

	result, filter_result = asyncio.run(pipeline.run("", top_k=10))

	assert result.expansion.original_query == ""
	assert result.expansion.expanded_terms == []
	assert result.candidate_reviews == []
	assert result.filtered_reviews == []
	assert result.llm_summary != ""
	assert filter_result.topic_distribution == {}
