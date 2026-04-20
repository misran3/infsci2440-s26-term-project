"""Score ordering heuristic tests."""

import pytest

from src.judge.heuristics import check_recall_improvement, check_score_ordering
from src.search import BeamSearchExpander, TFIDFRetriever


@pytest.mark.asyncio
async def test_score_ordering(judge_collector, reviews):
    """Verify score ordering is monotonically decreasing."""
    retriever = TFIDFRetriever(reviews)
    retriever.fit()

    results = retriever.retrieve(["game", "fun"], top_k=10)
    scores = [r.tfidf_score for r in results]

    heuristic = check_score_ordering("game fun", scores)
    judge_collector.add_test(heuristic)


@pytest.mark.asyncio
async def test_recall_improvement(judge_collector, reviews):
    """Verify expansion doesn't decrease recall."""
    expander = BeamSearchExpander(beam_width=3, max_depth=2)
    retriever = TFIDFRetriever(reviews)
    retriever.fit()

    query = "problem"
    expansion = expander.expand(query)

    base_results = retriever.retrieve([query], top_k=50)
    expanded_results = retriever.retrieve(expansion.expanded_terms, top_k=50)

    heuristic = check_recall_improvement(
        query=query,
        base_count=len(base_results),
        expanded_count=len(expanded_results),
    )
    judge_collector.add_test(heuristic)
