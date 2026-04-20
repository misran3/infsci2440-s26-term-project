"""Tests for TF-IDF retrieval."""

import pytest
from src.loaders.structures import Review
from src.search.tfidf_retriever import TFIDFRetriever


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
