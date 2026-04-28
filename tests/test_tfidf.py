"""Tests for TF-IDF retrieval."""

import os
import tempfile

import joblib
import pytest

from src.loaders.structures import Review
from src.search.tfidf_retriever import (
    TFIDFRetriever,
    load_model,
    load_retriever,
    save_model,
)


def test_retrieve_returns_relevant():
    """Retrieval should return reviews matching query terms."""
    corpus = [
        Review("1", "shipping was fast and reliable", 5, "Good", "A"),
        Review("2", "product quality is poor", 1, "Bad", "B"),
        Review("3", "delivery took forever", 2, "Slow", "C"),
    ]
    retriever = TFIDFRetriever(corpus)
    retriever.fit()

    results = retriever.retrieve(["shipping", "delivery"], top_k=2)

    assert len(results) == 2
    review_ids = [r.review_id for r in results]
    assert "1" in review_ids  # Contains "shipping"
    assert "3" in review_ids  # Contains "delivery"
    assert "2" not in review_ids  # No shipping/delivery terms


def test_empty_corpus():
    """Empty corpus should not crash."""
    retriever = TFIDFRetriever([])
    retriever.fit()

    results = retriever.retrieve(["shipping"])
    assert results == []


def test_empty_terms():
    """Empty terms should return empty results."""
    corpus = [
        Review("1", "shipping was fast", 5, "Good", "A"),
    ]
    retriever = TFIDFRetriever(corpus)
    retriever.fit()

    results = retriever.retrieve([])
    assert results == []


def test_no_matches():
    """No matching terms should return empty results."""
    corpus = [
        Review("1", "shipping was fast", 5, "Good", "A"),
        Review("2", "delivery is slow", 2, "Bad", "B"),
    ]
    retriever = TFIDFRetriever(corpus)
    retriever.fit()

    results = retriever.retrieve(["xyznonexistent"])
    assert results == []


def test_top_k_larger_than_corpus():
    """top_k larger than corpus should return all matches."""
    corpus = [
        Review("1", "shipping was fast", 5, "Good", "A"),
        Review("2", "shipping is reliable", 4, "OK", "B"),
    ]
    retriever = TFIDFRetriever(corpus)
    retriever.fit()

    results = retriever.retrieve(["shipping"], top_k=100)
    assert len(results) == 2


def test_tfidf_scores_populated():
    """Every returned review should have tfidf_score > 0."""
    corpus = [
        Review("1", "shipping was fast", 5, "Good", "A"),
        Review("2", "delivery took time", 3, "OK", "B"),
    ]
    retriever = TFIDFRetriever(corpus)
    retriever.fit()

    results = retriever.retrieve(["shipping", "delivery"])
    for review in results:
        assert review.tfidf_score is not None
        assert review.tfidf_score > 0


def test_results_sorted_descending():
    """Results should be sorted by score descending."""
    corpus = [
        Review("1", "shipping", 5, "A", "X"),
        Review("2", "shipping shipping shipping", 5, "B", "X"),
        Review("3", "shipping fast", 5, "C", "X"),
    ]
    retriever = TFIDFRetriever(corpus)
    retriever.fit()

    results = retriever.retrieve(["shipping"])
    scores = [r.tfidf_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_get_matching_terms_accuracy():
    """get_matching_terms should correctly identify matched terms."""
    corpus = [
        Review("1", "shipping was fast and delivery reliable", 5, "Good", "A"),
    ]
    retriever = TFIDFRetriever(corpus)
    retriever.fit()

    review = corpus[0]
    matching = retriever.get_matching_terms(review, ["shipping", "delivery", "price"])
    assert "shipping" in matching
    assert "delivery" in matching
    assert "price" not in matching


def test_save_and_load_model():
    """Model should round-trip through save/load."""
    corpus = [
        Review("1", "shipping was fast", 5, "Good", "A"),
        Review("2", "delivery is reliable", 4, "OK", "B"),
    ]
    retriever = TFIDFRetriever(corpus)
    retriever.fit()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model.pkl")
        save_model(retriever, path)

        loaded = load_model(path)
        assert "vectorizer" in loaded
        assert "tfidf_matrix" in loaded
        assert "corpus_ids" in loaded
        assert loaded["corpus_ids"] == ["1", "2"]


def test_save_and_load_model_with_metadata():
    """Model should save and load metadata."""
    corpus = [
        Review("1", "shipping was fast", 5, "Good", "A"),
        Review("2", "delivery is reliable", 4, "OK", "B"),
    ]
    retriever = TFIDFRetriever(corpus)
    retriever.fit()

    metadata = {
        "trained_at": "2026-04-28T10:00:00",
        "data_source": "test_data.csv",
        "corpus_size": 2,
        "params": {"max_features": 10000, "ngram_range": [1, 2]},
        "metrics": {"vocabulary_size": 5},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model.pkl")
        save_model(retriever, path, metadata=metadata)

        loaded = load_model(path)
        assert "metadata" in loaded
        assert loaded["metadata"]["corpus_size"] == 2
        assert loaded["metadata"]["trained_at"] == "2026-04-28T10:00:00"


def test_load_model_without_metadata_returns_empty_metadata():
    """Old model files without metadata should load with empty metadata dict."""
    corpus = [
        Review("1", "shipping was fast", 5, "Good", "A"),
    ]
    retriever = TFIDFRetriever(corpus)
    retriever.fit()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "old_model.pkl")
        # Save in old format (without metadata parameter)
        joblib.dump(
            {
                "vectorizer": retriever.vectorizer,
                "tfidf_matrix": retriever.tfidf_matrix,
                "corpus_ids": [r.review_id for r in retriever.corpus],
            },
            path,
        )

        loaded = load_model(path)
        assert "metadata" in loaded
        assert loaded["metadata"] == {}


def test_load_retriever_restores_functionality():
    """load_retriever should restore a working retriever."""
    corpus = [
        Review("1", "shipping was fast", 5, "Good", "A"),
        Review("2", "delivery is reliable", 4, "OK", "B"),
    ]
    retriever = TFIDFRetriever(corpus)
    retriever.fit()

    metadata = {"corpus_size": 2, "trained_at": "2026-04-28T10:00:00"}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model.pkl")
        save_model(retriever, path, metadata=metadata)

        loaded_retriever, loaded_meta = load_retriever(path, corpus)

        # Verify retriever works
        results = loaded_retriever.retrieve(["shipping"], top_k=2)
        assert len(results) >= 1
        assert results[0].review_id == "1"

        # Verify metadata
        assert loaded_meta["corpus_size"] == 2
