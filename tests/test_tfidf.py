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
