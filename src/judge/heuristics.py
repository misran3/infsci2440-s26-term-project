"""Fast heuristic checks for search evaluation."""

from src.judge.models import HeuristicTest


def check_recall_improvement(
    query: str,
    base_count: int,
    expanded_count: int,
) -> HeuristicTest:
    """Check that expansion doesn't decrease recall."""
    return HeuristicTest(
        name="recall_improvement",
        query=query,
        passed=expanded_count >= base_count,
        details={
            "base_results": base_count,
            "expanded_results": expanded_count,
            "delta": expanded_count - base_count,
        },
    )


def check_score_ordering(
    query: str,
    scores: list[float],
) -> HeuristicTest:
    """Check that scores are monotonically decreasing."""
    is_ordered = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    return HeuristicTest(
        name="score_ordering",
        query=query,
        passed=is_ordered,
        details={
            "scores": scores,
            "is_monotonic": is_ordered,
        },
    )


def check_expansion_not_empty(
    query: str,
    expanded_terms: list[str],
) -> HeuristicTest:
    """Check that expansion returns at least the original term."""
    return HeuristicTest(
        name="expansion_not_empty",
        query=query,
        passed=len(expanded_terms) > 0,
        details={
            "term_count": len(expanded_terms),
            "terms": expanded_terms,
        },
    )
