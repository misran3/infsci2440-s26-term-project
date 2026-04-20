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
