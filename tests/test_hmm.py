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
	"""fit computes global transitions with expected keys."""
	hmm = HMMSentiment()
	reviews = [
		Review("1", "Good start. Then bad.", 2, "Mixed", "A"),
		Review("2", "Terrible initially. Now better.", 3, "Mixed", "B"),
	]

	hmm.fit(reviews)

	required_keys = {
		"pos_to_pos",
		"pos_to_neg",
		"neg_to_pos",
		"neg_to_neg",
		"neu_to_pos",
		"neu_to_neg",
		"neu_to_neu",
	}
	assert required_keys.issubset(set(hmm.global_transitions.keys()))
