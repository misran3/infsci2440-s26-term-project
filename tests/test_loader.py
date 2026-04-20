"""Tests for data loader module."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.loaders.structures import Review, Topic


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing."""
    data = {
        "review_id": ["R001", "R002", "R003"],
        "text": ["Great software!", "Terrible bugs.", "Average product."],
        "rating": [5, 1, 3],
        "title": ["Love it", "Hate it", "Meh"],
        "product_id": ["P001", "P002", "P003"],
        "sentences": [
            json.dumps(["Great software!"]),
            json.dumps(["Terrible bugs."]),
            json.dumps(["Average product."]),
        ],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_reviews.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def labeled_csv(tmp_path):
    """Create a labeled CSV file for testing."""
    data = {
        "review_id": ["R001", "R002"],
        "text": ["Software crashes often.", "Easy to use interface."],
        "rating": [2, 5],
        "title": ["Buggy", "Great"],
        "product_id": ["P001", "P002"],
        "sentences": [
            json.dumps(["Software crashes often."]),
            json.dumps(["Easy to use interface."]),
        ],
        "topic_label": ["performance", "usability"],
        "confidence_score": [3, 2],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "labeled_reviews.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_load_reviews_returns_list_of_reviews(sample_csv):
    """load_reviews should return a list of Review objects."""
    from src.loaders.loader import load_reviews

    reviews = load_reviews(path=sample_csv)

    assert isinstance(reviews, list)
    assert len(reviews) == 3
    assert all(isinstance(r, Review) for r in reviews)


def test_load_reviews_parses_sentences(sample_csv):
    """load_reviews should parse JSON sentences column."""
    from src.loaders.loader import load_reviews

    reviews = load_reviews(path=sample_csv)

    assert reviews[0].sentences == ["Great software!"]
    assert reviews[1].sentences == ["Terrible bugs."]


def test_load_reviews_with_limit(sample_csv):
    """load_reviews should respect limit parameter."""
    from src.loaders.loader import load_reviews

    reviews = load_reviews(path=sample_csv, limit=2)

    assert len(reviews) == 2


def test_load_labeled_reviews_returns_reviews_and_labels(labeled_csv, monkeypatch):
    """load_labeled_reviews should return tuple of reviews and labels."""
    from src.loaders import loader
    from src.config import DataConfig

    # Monkeypatch DATA to use our test file
    test_config = DataConfig(
        labeled_reviews=labeled_csv,
        curated_labels=labeled_csv.parent / "nonexistent.csv",
        raw_reviews=labeled_csv,
        clean_reviews=labeled_csv,
        main_corpus=labeled_csv,
        sample_reviews=labeled_csv,
    )
    monkeypatch.setattr(loader, "DATA", test_config)

    reviews, labels = loader.load_labeled_reviews(use_curated=False)

    assert len(reviews) == 2
    assert len(labels) == 2
    assert labels == ["performance", "usability"]


def test_get_corpus_stats(sample_csv):
    """get_corpus_stats should return statistics dict."""
    from src.loaders.loader import get_corpus_stats

    stats = get_corpus_stats(path=sample_csv)

    assert stats["total_reviews"] == 3
    assert "rating_distribution" in stats
    assert stats["has_sentences"] is True
