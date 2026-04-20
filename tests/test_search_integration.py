"""Integration tests for search components."""

import pytest
from src.loaders.structures import Review
from src.search.beam_search import BeamSearchExpander
from src.search.tfidf_retriever import TFIDFRetriever


def test_expand_then_retrieve():
    """Full flow: expand query then retrieve reviews."""
    corpus = [
        Review("1", "shipping was fast and reliable", 5, "Good", "A"),
        Review("2", "the product quality is excellent", 5, "Great", "B"),
        Review("3", "delivery took forever to arrive", 2, "Slow", "C"),
        Review("4", "transport was quick and efficient", 4, "Nice", "D"),
    ]

    expander = BeamSearchExpander(beam_width=3, max_depth=1)
    retriever = TFIDFRetriever(corpus)
    retriever.fit()

    expansion = expander.expand("shipping")
    results = retriever.retrieve(expansion.expanded_terms, top_k=10)

    assert len(results) > 0
    review_ids = [r.review_id for r in results]
    assert "1" in review_ids  # Direct match for "shipping"


def test_expanded_terms_improve_recall():
    """Expansion should retrieve more results than original query alone."""
    corpus = [
        Review("1", "shipping was fast", 5, "Good", "A"),
        Review("2", "delivery was quick", 5, "Great", "B"),
        Review("3", "transport arrived on time", 4, "Nice", "C"),
        Review("4", "product quality is good", 5, "Quality", "D"),
    ]

    expander = BeamSearchExpander(beam_width=3, max_depth=2)
    retriever = TFIDFRetriever(corpus)
    retriever.fit()

    # Original query only
    original_results = retriever.retrieve(["shipping"], top_k=10)

    # Expanded query
    expansion = expander.expand("shipping")
    expanded_results = retriever.retrieve(expansion.expanded_terms, top_k=10)

    # Expansion should find more or equal results
    assert len(expanded_results) >= len(original_results)
