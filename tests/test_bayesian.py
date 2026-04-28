"""Tests for Bayesian network reasoning."""

import tempfile
from pathlib import Path

import pytest

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


def test_has_pgmpy_model_after_fit():
	"""BayesianNetwork should have a pgmpy model after fitting."""
	from pgmpy.models import DiscreteBayesianNetwork as PgmpyBN

	bn = BayesianNetwork()
	bn.fit(_reviews())

	assert hasattr(bn, "model")
	assert isinstance(bn.model, PgmpyBN)


def test_save_raises_on_unfitted_model():
	"""save() should raise RuntimeError if model is not fitted."""
	bn = BayesianNetwork()

	with tempfile.NamedTemporaryFile(suffix=".joblib") as f:
		with pytest.raises(RuntimeError, match="Cannot save unfitted BayesianNetwork"):
			bn.save(f.name)


def test_save_load_roundtrip_preserves_inference():
	"""save/load round-trip should preserve inference results."""
	bn = BayesianNetwork()
	bn.fit(_reviews())

	# Get inference results before save
	insights_before = bn.infer([], "performance")

	with tempfile.TemporaryDirectory() as tmpdir:
		path = Path(tmpdir) / "model.joblib"
		bn.save(path)

		# Load the model
		loaded_bn = BayesianNetwork.load(path)

		# Get inference results after load
		insights_after = loaded_bn.infer([], "performance")

	# Verify results match
	assert insights_after.topic == insights_before.topic
	assert insights_after.p_positive_given_topic == pytest.approx(
		insights_before.p_positive_given_topic, rel=1e-6
	)
	assert insights_after.p_negative_given_topic == pytest.approx(
		insights_before.p_negative_given_topic, rel=1e-6
	)
	assert insights_after.p_high_rating_given_positive == pytest.approx(
		insights_before.p_high_rating_given_positive, rel=1e-6
	)
	assert insights_after.p_low_rating_given_negative == pytest.approx(
		insights_before.p_low_rating_given_negative, rel=1e-6
	)


def test_load_restores_fitted_state():
	"""load() should restore _is_fitted state."""
	bn = BayesianNetwork()
	bn.fit(_reviews())

	with tempfile.TemporaryDirectory() as tmpdir:
		path = Path(tmpdir) / "model.joblib"
		bn.save(path)

		loaded_bn = BayesianNetwork.load(path)

	assert loaded_bn._is_fitted is True
	assert loaded_bn._inference is not None


def test_load_raises_on_invalid_model_type():
	"""load() should raise ValueError if loaded object is not a pgmpy model."""
	import joblib

	with tempfile.TemporaryDirectory() as tmpdir:
		path = Path(tmpdir) / "invalid.joblib"
		# Save something that is not a BayesianNetwork
		joblib.dump({"not": "a model"}, path)

		with pytest.raises(ValueError, match="Invalid model type"):
			BayesianNetwork.load(path)


def test_save_creates_parent_directories():
	"""save() should create parent directories if they don't exist."""
	bn = BayesianNetwork()
	bn.fit(_reviews())

	with tempfile.TemporaryDirectory() as tmpdir:
		path = Path(tmpdir) / "nested" / "dir" / "model.joblib"
		bn.save(path)

		assert path.exists()


def test_save_and_load_with_metadata():
	"""Saved model should include and restore metadata."""
	bn = BayesianNetwork()
	bn.fit(_reviews())

	metadata = {
		"trained_at": "2026-04-28T10:00:00",
		"data_source": "curated_labels.csv",
		"corpus_size": 4,
		"params": {"structure": [["topic", "sentiment"], ["sentiment", "rating_category"]]},
		"metrics": {"n_cpds": 3},
	}

	with tempfile.TemporaryDirectory() as tmpdir:
		path = Path(tmpdir) / "model.pkl"
		bn.save(path, metadata=metadata)

		loaded_bn = BayesianNetwork.load(path)
		assert hasattr(loaded_bn, "metadata")
		assert loaded_bn.metadata["corpus_size"] == 4
		assert loaded_bn.metadata["trained_at"] == "2026-04-28T10:00:00"


def test_load_without_metadata_returns_empty_metadata():
	"""Old model files without metadata should load with empty metadata dict."""
	import joblib

	bn = BayesianNetwork()
	bn.fit(_reviews())

	with tempfile.TemporaryDirectory() as tmpdir:
		path = Path(tmpdir) / "old_model.pkl"
		# Save in old format (raw model, no metadata wrapper)
		joblib.dump(bn.model, path)

		loaded_bn = BayesianNetwork.load(path)
		assert hasattr(loaded_bn, "metadata")
		assert loaded_bn.metadata == {}
