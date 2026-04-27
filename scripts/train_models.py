"""Train and save all project models."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from src.classification.naive_bayes import TopicClassifier
from src.config import MODELS, MODELS_DIR
from src.loaders.loader import load_labeled_reviews, load_reviews
from src.loaders.structures import Topic
from src.reasoning.bayesian_network import BayesianNetwork
from src.reasoning.hmm_sentiment import HMMSentiment
from src.search.tfidf_retriever import TFIDFRetriever, save_model as save_tfidf_model


def _build_reviews_with_topics() -> list:
	"""Load labeled reviews and attach topics for probabilistic modules."""
	reviews, labels = load_labeled_reviews(use_curated=True)
	for review, label in zip(reviews, labels):
		try:
			review.topic = review.topic or Topic(label)
		except ValueError:
			review.topic = Topic.OTHER
	return reviews


def train_all(output_dir: Path | None = None) -> None:
	"""Train all models required by the pipeline."""
	output_root = output_dir or MODELS_DIR
	output_root.mkdir(parents=True, exist_ok=True)

	print("=" * 60)
	print("TRAINING ALL MODELS")
	print("=" * 60)

	# 1. TF-IDF model on main corpus
	print("\n[1/4] Training TF-IDF retriever...")
	corpus = load_reviews(path=None, sample=False)
	retriever = TFIDFRetriever(corpus)
	retriever.fit()
	tfidf_path = output_root / MODELS.tfidf_vectorizer.name
	save_tfidf_model(retriever, str(tfidf_path))
	print(f"Saved TF-IDF model to {tfidf_path}")

	# 2. Naive Bayes classifier on labeled data
	print("\n[2/4] Training Naive Bayes classifier...")
	nb_reviews, nb_labels = load_labeled_reviews(use_curated=True)
	classifier = TopicClassifier()
	metrics = classifier.fit(nb_reviews, nb_labels)
	nb_path = output_root / MODELS.naive_bayes.name
	classifier.save(str(nb_path))
	print(f"Saved Naive Bayes model to {nb_path}")
	print(f"Training samples: {metrics['n_samples']}, features: {metrics['n_features']}")

	# 3. Bayesian reasoning model state
	print("\n[3/4] Fitting Bayesian model...")
	bayesian_reviews = _build_reviews_with_topics()
	bayesian = BayesianNetwork()
	bayesian.fit(bayesian_reviews)
	bayesian_path = output_root / MODELS.bayesian_network.name
	joblib.dump({"fitted_reviews": bayesian_reviews}, bayesian_path)
	print(f"Saved Bayesian state to {bayesian_path}")

	# 4. HMM sentiment transition model state
	print("\n[4/4] Fitting HMM sentiment model...")
	hmm_reviews = load_reviews(path=None, sample=False, limit=10000)
	hmm = HMMSentiment()
	hmm.fit(hmm_reviews)
	hmm_path = output_root / MODELS.hmm_model.name
	joblib.dump({"global_transitions": hmm.global_transitions}, hmm_path)
	print(f"Saved HMM state to {hmm_path}")

	print("\nDone.")


def main() -> None:
	"""CLI entry point for training all models."""
	parser = argparse.ArgumentParser(description="Train all survey-analysis models")
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=MODELS_DIR,
		help="Directory to write model artifacts",
	)
	args = parser.parse_args()
	train_all(args.output_dir)


if __name__ == "__main__":
	main()
