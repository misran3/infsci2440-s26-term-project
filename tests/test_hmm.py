"""Tests for HMM-inspired sentiment sequence analysis."""

import tempfile
from pathlib import Path

import joblib
import pytest

from src.loaders.structures import Review, Sentiment
from src.reasoning.hmm_sentiment import HMMSentiment


def _reviews() -> list[Review]:
    """Helper to create test reviews."""
    return [
        Review("1", "Great product. It works well. Very happy.", 5, "Good", "A"),
        Review("2", "Terrible. Crashed. Waste of money.", 1, "Bad", "B"),
        Review("3", "Okay start. Got worse. Then better.", 3, "Mixed", "C"),
    ]


def test_analyze_returns_sequence_per_review():
	"""analyze returns one sequence object per review."""
	hmm = HMMSentiment()
	reviews = [
		Review(
			"1",
			"Great product. It later crashed.",
			2,
			"Mixed",
			"A",
		)
	]

	sequences = hmm.analyze(reviews)

	assert len(sequences) == 1
	assert sequences[0].review_id == "1"
	assert len(sequences[0].sentences) >= 1
	assert len(sequences[0].sentiment_states) == len(sequences[0].sentences)
	assert all(isinstance(s, Sentiment) for s in sequences[0].sentiment_states)


def test_analyze_uses_pre_tokenized_sentences_when_present():
	"""When review.sentences exists, analyzer should preserve it."""
	hmm = HMMSentiment()
	review = Review(
		"2",
		"ignored",
		4,
		"Title",
		"B",
		sentences=["Very easy to use.", "Highly recommended."],
	)

	sequence = hmm.analyze([review])[0]

	assert sequence.sentences == review.sentences
	assert len(sequence.sentiment_states) == 2


def test_fit_updates_global_transitions():
	"""fit computes global transitions with all 9 keys."""
	hmm = HMMSentiment()
	reviews = [
		Review("1", "Good start. Then bad. Finally okay.", 2, "Mixed", "A"),
		Review("2", "Terrible initially. Now better. Great ending.", 3, "Mixed", "B"),
	]

	sequences = hmm.analyze(reviews)

	required_keys = {
		"pos_to_pos", "pos_to_neg", "pos_to_neu",
		"neg_to_pos", "neg_to_neg", "neg_to_neu",
		"neu_to_pos", "neu_to_neg", "neu_to_neu",
	}
	for seq in sequences:
		assert required_keys == set(seq.transitions.keys())


def test_has_hmmlearn_model_after_fit():
	"""HMMSentiment should have an hmmlearn model after fitting."""
	from hmmlearn.hmm import CategoricalHMM

	hmm = HMMSentiment()
	hmm.fit(_reviews())

	assert hasattr(hmm, "model")
	assert isinstance(hmm.model, CategoricalHMM)


def test_save_raises_on_unfitted_model():
	"""save() should raise RuntimeError if model is not fitted."""
	hmm = HMMSentiment()

	with tempfile.NamedTemporaryFile(suffix=".joblib") as f:
		with pytest.raises(RuntimeError, match="Cannot save unfitted HMMSentiment"):
			hmm.save(f.name)


def test_save_load_roundtrip_preserves_inference():
	"""save/load round-trip should preserve inference results."""
	hmm = HMMSentiment()
	hmm.fit(_reviews())

	# Get analyze results before save
	test_review = [Review("test", "Good start. Bad middle. Okay end.", 3, "Mixed", "X")]
	results_before = hmm.analyze(test_review)

	with tempfile.TemporaryDirectory() as tmpdir:
		path = Path(tmpdir) / "model.joblib"
		hmm.save(path)

		# Load the model
		loaded_hmm = HMMSentiment.load(path)

		# Get analyze results after load
		results_after = loaded_hmm.analyze(test_review)

	# Verify results match
	assert results_after[0].review_id == results_before[0].review_id
	assert results_after[0].sentences == results_before[0].sentences
	assert results_after[0].sentiment_states == results_before[0].sentiment_states


def test_load_restores_fitted_state():
	"""load() should restore _is_fitted state."""
	hmm = HMMSentiment()
	hmm.fit(_reviews())

	with tempfile.TemporaryDirectory() as tmpdir:
		path = Path(tmpdir) / "model.joblib"
		hmm.save(path)

		loaded_hmm = HMMSentiment.load(path)

	assert loaded_hmm._is_fitted is True


def test_load_raises_on_invalid_model_type():
	"""load() should raise ValueError if loaded object is not a CategoricalHMM."""
	import joblib

	with tempfile.TemporaryDirectory() as tmpdir:
		path = Path(tmpdir) / "invalid.joblib"
		# Save something that is not a CategoricalHMM
		joblib.dump({"not": "a model"}, path)

		with pytest.raises(ValueError, match="Invalid model type"):
			HMMSentiment.load(path)


def test_save_creates_parent_directories():
	"""save() should create parent directories if they don't exist."""
	hmm = HMMSentiment()
	hmm.fit(_reviews())

	with tempfile.TemporaryDirectory() as tmpdir:
		path = Path(tmpdir) / "nested" / "dir" / "model.joblib"
		hmm.save(path)

		assert path.exists()


def test_save_and_load_with_metadata():
	"""Saved model should include and restore metadata."""
	hmm = HMMSentiment()
	hmm.fit(_reviews())

	metadata = {
		"trained_at": "2026-04-28T10:00:00",
		"data_source": "clean_reviews.csv",
		"corpus_size": 3,
		"params": {"n_components": 3, "n_iter": 100},
		"metrics": {"converged": True},
	}

	with tempfile.TemporaryDirectory() as tmpdir:
		path = Path(tmpdir) / "model.pkl"
		hmm.save(path, metadata=metadata)

		loaded_hmm = HMMSentiment.load(path)
		assert hasattr(loaded_hmm, "metadata")
		assert loaded_hmm.metadata["corpus_size"] == 3
		assert loaded_hmm.metadata["trained_at"] == "2026-04-28T10:00:00"


def test_load_without_metadata_returns_empty_metadata():
	"""Old model files without metadata should load with empty metadata dict."""
	hmm = HMMSentiment()
	hmm.fit(_reviews())

	with tempfile.TemporaryDirectory() as tmpdir:
		path = Path(tmpdir) / "old_model.pkl"
		# Save in old format (raw model, no metadata wrapper)
		joblib.dump(hmm.model, path)

		loaded_hmm = HMMSentiment.load(path)
		assert hasattr(loaded_hmm, "metadata")
		assert loaded_hmm.metadata == {}
