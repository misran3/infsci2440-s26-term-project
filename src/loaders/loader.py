"""Load reviews from CSV files."""

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import DATA
from src.loaders.structures import Review, Topic


def load_reviews(
    path: Optional[Path] = None,
    sample: bool = False,
    limit: Optional[int] = None,
) -> list[Review]:
    """
    Load reviews from CSV.

    Args:
        path: Custom path. Defaults to main corpus.
        sample: If True, load sample_reviews.csv instead.
        limit: Max reviews to load (for testing).

    Returns:
        List of Review objects.
    """
    if path is None:
        path = DATA.sample_reviews if sample else DATA.main_corpus

    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)

    reviews = []
    for _, row in df.iterrows():
        # Parse pre-tokenized sentences if available
        sentences = None
        if "sentences" in row and pd.notna(row["sentences"]):
            try:
                sentences = json.loads(row["sentences"])
            except json.JSONDecodeError:
                pass

        # Parse topic if available
        topic = None
        if "topic" in row and pd.notna(row["topic"]):
            try:
                topic = Topic(row["topic"])
            except ValueError:
                pass

        review = Review(
            review_id=str(row["review_id"]),
            text=str(row["text"]),
            rating=int(row["rating"]),
            title=str(row.get("title", "")),
            product_id=str(row.get("product_id", "")),
            sentences=sentences,
            topic=topic,
        )
        reviews.append(review)

    return reviews


def load_labeled_reviews(use_curated: bool = True) -> tuple[list[Review], list[str]]:
    """
    Load reviews with topic labels for Naive Bayes training.

    Args:
        use_curated: If True, prefer curated_labels.csv when available.

    Returns:
        Tuple of (reviews, labels).
    """
    # Prefer curated labels if available
    if use_curated and DATA.curated_labels.exists():
        df = pd.read_csv(DATA.curated_labels)
        # Use verified_topic if present, else fall back to auto_topic
        if "verified_topic" in df.columns:
            df = df[df["verified_topic"].notna()]
            labels = df["verified_topic"].tolist()
        else:
            df = df[df["auto_topic"].notna()]
            labels = df["auto_topic"].tolist()
    else:
        df = pd.read_csv(DATA.labeled_reviews)
        labels = df["topic_label"].tolist()

    # Build reviews list from the dataframe
    reviews = []
    for _, row in df.iterrows():
        sentences = None
        if "sentences" in row and pd.notna(row["sentences"]):
            try:
                sentences = json.loads(row["sentences"])
            except json.JSONDecodeError:
                pass

        review = Review(
            review_id=str(row["review_id"]),
            text=str(row["text"]),
            rating=int(row["rating"]),
            title=str(row.get("title", "")),
            product_id=str(row.get("product_id", "")),
            sentences=sentences,
        )
        reviews.append(review)

    return reviews, labels


def get_corpus_stats(path: Optional[Path] = None) -> dict:
    """Get statistics about a corpus file."""
    if path is None:
        path = DATA.main_corpus

    df = pd.read_csv(path)
    return {
        "total_reviews": len(df),
        "rating_distribution": df["rating"].value_counts().to_dict(),
        "avg_text_length": df["text"].str.len().mean(),
        "has_sentences": "sentences" in df.columns,
    }
