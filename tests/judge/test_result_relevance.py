"""Result relevance evaluation tests."""

import pytest

from src.judge.llm_judge import evaluate_result_relevance
from src.judge.models import ResultRelevanceTest
from src.search import BeamSearchExpander, TFIDFRetriever

QUERIES = ["problem", "good quality"]


@pytest.mark.asyncio
async def test_result_relevance(judge_collector, reviews):
    """Evaluate result relevance for common queries."""
    expander = BeamSearchExpander(beam_width=3, max_depth=2)
    retriever = TFIDFRetriever(reviews)
    retriever.fit()

    for query in QUERIES:
        expansion = expander.expand(query)
        results = retriever.retrieve(expansion.expanded_terms, top_k=5)

        formatted = [{"review_id": r.review_id, "text": r.text} for r in results]
        judgment = await evaluate_result_relevance(query, formatted)
        test_result = ResultRelevanceTest(
            query=query,
            results=judgment.judgments,
        )
        judge_collector.add_test(test_result)
