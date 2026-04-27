"""Additional tests for Naive Bayes classifier."""

from pathlib import Path

from src.classification.naive_bayes import TopicClassifier
from src.loaders.structures import Review, Topic


def _training_set() -> tuple[list[Review], list[str]]:
	reviews = [
		Review("1", "Crashes and freezes", 1, "Bad", "A"),
		Review("2", "Easy interface and simple workflow", 5, "Great", "B"),
		Review("3", "Too expensive subscription", 2, "Pricey", "C"),
	]
	labels = ["performance", "usability", "pricing"]
	return reviews, labels


def test_save_and_load_round_trip(tmp_path: Path):
	"""Saved classifier should load and remain usable."""
	reviews, labels = _training_set()
	classifier = TopicClassifier()
	classifier.fit(reviews, labels)

	model_path = tmp_path / "naive_bayes.pkl"
	classifier.save(str(model_path))

	loaded = TopicClassifier.load(str(model_path))
	pred = loaded.predict([Review("99", "app freezes", 1, "T", "P")])[0]
	assert pred.predicted_topic == Topic.PERFORMANCE


def test_detect_topic_from_query_defaults_to_other():
	"""Unknown query terms should map to 'other'."""
	classifier = TopicClassifier()
	detected = classifier.detect_topic_from_query("quantum entanglement in photons")
	assert detected == "other"
