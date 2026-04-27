"""Tests for Bayesian network reasoning."""

from src.loaders.structures import Review, Sentiment, Topic
from src.reasoning.bayesian_network import BayesianNetwork


def _reviews() -> list[Review]:
	return [
		Review("1", "Crashes often", 1, "Bad", "A", topic=Topic.PERFORMANCE, sentiment=Sentiment.NEGATIVE),
		Review("2", "Fast and stable", 5, "Great", "B", topic=Topic.PERFORMANCE, sentiment=Sentiment.POSITIVE),
		Review("3", "Slow startup", 2, "Poor", "C", topic=Topic.PERFORMANCE, sentiment=Sentiment.NEGATIVE),
		Review("4", "Easy to use", 5, "Nice", "D", topic=Topic.USABILITY, sentiment=Sentiment.POSITIVE),
	]


def test_infer_returns_valid_probabilities():
	"""infer returns probabilities in [0, 1] for known topic."""
	bn = BayesianNetwork()
	insights = bn.infer(_reviews(), "performance")

	assert insights.topic == Topic.PERFORMANCE
	assert 0.0 <= insights.p_positive_given_topic <= 1.0
	assert 0.0 <= insights.p_negative_given_topic <= 1.0
	assert 0.0 <= insights.p_high_rating_given_positive <= 1.0
	assert 0.0 <= insights.p_low_rating_given_negative <= 1.0


def test_infer_unknown_topic_falls_back_safely():
	"""Unknown topic string should map to OTHER and still return insights."""
	bn = BayesianNetwork()
	insights = bn.infer(_reviews(), "not-a-real-topic")

	assert insights.topic == Topic.OTHER
	assert 0.0 <= insights.p_positive_given_topic <= 1.0
	assert 0.0 <= insights.p_negative_given_topic <= 1.0


def test_fit_then_infer_without_input_reviews_uses_fitted_data():
	"""infer should use fitted corpus when call-time reviews are empty."""
	bn = BayesianNetwork()
	bn.fit(_reviews())

	insights = bn.infer([], "usability")

	assert insights.topic == Topic.USABILITY
	assert insights.p_positive_given_topic >= insights.p_negative_given_topic
