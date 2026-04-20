"""LLM-based semantic evaluation for search quality."""

from src.judge.models import (
    ResultRelevanceBatchResult,
    TermRelevanceBatchResult,
)
from src.llm import get_agent

TERM_RELEVANCE_PROMPT = """You evaluate whether expanded search terms are semantically relevant
to the original query in the context of software product reviews.

Consider: Would a user searching for the original query also want results containing this term?
Irrelevant terms include: wrong word senses (e.g., "bug" as insect), unrelated synonyms,
or terms that would introduce noise in software review search results."""

RESULT_RELEVANCE_PROMPT = """You evaluate whether retrieved software reviews are relevant
to the user's search query.

Consider: Does this review discuss topics related to the query? A review is relevant if
it addresses the concept the user was searching for, even if it doesn't contain the exact term."""


async def evaluate_term_relevance(
    query: str, terms: list[str]
) -> TermRelevanceBatchResult:
    """Evaluate semantic relevance of expanded terms."""
    agent = get_agent(TermRelevanceBatchResult, TERM_RELEVANCE_PROMPT)
    result = await agent.run(f"Query: {query}\nExpanded terms to evaluate: {terms}")
    return result.output


async def evaluate_result_relevance(
    query: str,
    results: list[dict],
) -> ResultRelevanceBatchResult:
    """Evaluate relevance of retrieved reviews."""
    agent = get_agent(ResultRelevanceBatchResult, RESULT_RELEVANCE_PROMPT)
    formatted = "\n".join(f"- {r['review_id']}: {r['text'][:300]}" for r in results)
    result = await agent.run(f"Query: {query}\nRetrieved reviews:\n{formatted}")
    return result.output
