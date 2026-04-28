"""Train the Naive Bayes topic classifier."""

import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

from src.classification.naive_bayes import TopicClassifier
from src.config import DATA, MODELS
from src.loaders.structures import Review


def load_training_data(dataset: str) -> tuple[list[Review], list[str]]:
    """Load training data based on dataset choice."""
    df = pd.read_csv(DATA.labeled_reviews)

    if dataset == "sample":
        df = df.head(5000)
        print(f"Loaded sample dataset ({len(df)} reviews)")
    else:
        print(f"Loaded full dataset ({len(df)} reviews)")

    reviews = []
    labels = []

    for _, row in df.iterrows():
        review = Review(
            review_id=str(row["review_id"]),
            text=str(row["text"]),
            rating=int(row["rating"]),
            title=str(row.get("title", "")),
            product_id=str(row.get("product_id", "")),
        )
        reviews.append(review)
        labels.append(str(row["topic_label"]))

    return reviews, labels


def load_evaluation_data() -> tuple[list[Review], list[str]]:
    """Load curated gold set for evaluation."""
    df = pd.read_csv(DATA.curated_labels)

    reviews = []
    labels = []

    for _, row in df.iterrows():
        if pd.notna(row.get("verified_topic")):
            review = Review(
                review_id=str(row["review_id"]),
                text=str(row["text"]),
                rating=int(row["rating"]),
                title=str(row.get("title", "")),
                product_id=str(row.get("product_id", "")),
            )
            reviews.append(review)
            labels.append(str(row["verified_topic"]))

    print(f"Loaded evaluation set ({len(reviews)} gold-labeled reviews)")
    return reviews, labels


def main():
    parser = argparse.ArgumentParser(description="Train Naive Bayes classifier")
    parser.add_argument(
        "--dataset",
        choices=["sample", "full"],
        default="sample",
        help="Dataset to train on: sample (5K) or full",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(MODELS.naive_bayes),
        help="Output path for trained model",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate against curated gold set",
    )
    args = parser.parse_args()

    # Load training data
    print(f"\n{'='*60}")
    print("TRAINING NAIVE BAYES CLASSIFIER")
    print(f"{'='*60}\n")

    reviews, labels = load_training_data(args.dataset)

    # Train
    print("\nTraining...")
    classifier = TopicClassifier()

    start = time.time()
    metrics = classifier.fit(reviews, labels)
    elapsed = time.time() - start

    print(f"  - Vocabulary size: {metrics['n_features']}")
    print(f"  - Classes: {metrics['classes']}")
    print(f"  - Training time: {elapsed:.2f}s")

    # Save model with metadata
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "data_source": str(DATA.labeled_reviews),
        "corpus_size": metrics["n_samples"],
        "params": {
            "alpha": classifier.classifier.alpha,
            "max_features": classifier.vectorizer.max_features,
            "ngram_range": list(classifier.vectorizer.ngram_range),
        },
        "metrics": {
            "n_features": metrics["n_features"],
            "classes": [str(c) for c in metrics["classes"]],
        },
    }
    classifier.save(args.output, metadata=metadata)
    print(f"\nModel saved to {args.output}")

    # Evaluate
    if args.evaluate:
        print(f"\n{'='*60}")
        print("EVALUATION AGAINST GOLD SET")
        print(f"{'='*60}\n")

        eval_reviews, eval_labels = load_evaluation_data()
        predictions = classifier.predict(eval_reviews)

        pred_labels = [p.predicted_topic.value for p in predictions]

        accuracy = accuracy_score(eval_labels, pred_labels)
        print(f"Accuracy: {accuracy:.1%}\n")

        print("Classification Report:")
        print(classification_report(eval_labels, pred_labels, zero_division=0))


if __name__ == "__main__":
    main()
