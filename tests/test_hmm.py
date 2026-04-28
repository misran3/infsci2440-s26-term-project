"""Tests for HMM-inspired sentiment sequence analysis."""

from src.loaders.structures import Review, Sentiment
from src.reasoning.hmm_sentiment import HMMSentiment


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
	reviews = [
		Review("1", "Great product. It works well. Very happy.", 5, "Good", "A"),
		Review("2", "Terrible. Crashed. Waste of money.", 1, "Bad", "B"),
		Review("3", "Okay start. Got worse. Then better.", 3, "Mixed", "C"),
	]
	hmm.fit(reviews)

	assert hasattr(hmm, "model")
	assert isinstance(hmm.model, CategoricalHMM)
