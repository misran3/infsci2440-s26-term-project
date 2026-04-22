"""Term relevance evaluation tests."""

import pytest

from src.judge.heuristics import check_expansion_not_empty
from src.judge.llm_judge import evaluate_term_relevance
from src.judge.models import TermRelevanceTest
from src.search import BeamSearchExpander, TermFilter

QUERIES = ["bug", "crash", "good", "easy"]


@pytest.mark.asyncio
async def test_term_relevance(judge_collector):
    """Evaluate term relevance for common queries."""
    expander = BeamSearchExpander(beam_width=3, max_depth=2)
    term_filter = TermFilter(validate_credentials=False)

    for query in QUERIES:
        result = expander.expand(query)

        heuristic = check_expansion_not_empty(query, result.expanded_terms)
        judge_collector.add_test(heuristic)

        filtered_terms = await term_filter.filter(query, result.expanded_terms)

        judgment = await evaluate_term_relevance(query, filtered_terms)
        test_result = TermRelevanceTest(
            query=query,
            expanded_terms=filtered_terms,
            judgments=judgment.judgments,
        )
        judge_collector.add_test(test_result)
