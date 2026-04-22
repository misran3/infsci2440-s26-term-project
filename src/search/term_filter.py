"""LLM-based term filtering for search expansion."""

import json
import logging
from pathlib import Path

from pydantic import BaseModel

from src.config import DATA_DIR
from src.llm import get_agent

logger = logging.getLogger(__name__)

FILTER_PROMPT = """You filter search terms for software product reviews.
Given a list of candidate terms, return only those semantically relevant to
software reviews. Exclude wrong word senses (e.g., "bug" as insect) and
unrelated synonyms.

Return ONLY terms from the provided list that are relevant to software."""


class FilteredTerms(BaseModel):
    """LLM response model for filtered terms."""

    relevant_terms: list[str]


class TermFilter:
    """Filters expanded terms to those relevant for software reviews."""

    def __init__(self, cache_path: Path | None = None) -> None:
        self.cache_path = cache_path or DATA_DIR / "term_filter_cache.json"
        self.cache: dict[str, bool] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from disk."""
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text())
            logger.info(f"Loaded {len(self.cache)} cached terms")
        else:
            self.cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=2))

    async def filter(self, query: str, terms: list[str]) -> list[str]:
        """Filter terms to those relevant in software review context.

        Args:
            query: Original search query (always included in result).
            terms: Candidate terms from expansion.

        Returns:
            Filtered list of relevant terms.
        """
        query_terms = set(query.lower().split())
        terms_lower = [t.lower() for t in terms]

        cached_relevant = []
        uncached = []

        for term in terms_lower:
            if term in query_terms:
                cached_relevant.append(term)
            elif term in self.cache:
                if self.cache[term]:
                    cached_relevant.append(term)
            else:
                uncached.append(term)

        if not uncached:
            return cached_relevant

        agent = get_agent(
            output_type=FilteredTerms,
            system_prompt=FILTER_PROMPT,
            required=False,
        )

        if agent is None:
            logger.warning("LLM not available. Returning cached terms only.")
            return cached_relevant

        try:
            result = await agent.run(f"Query: {query}\nCandidate terms: {uncached}")
            llm_relevant = set(result.output.relevant_terms)

            for term in uncached:
                is_relevant = term in llm_relevant
                self.cache[term] = is_relevant
                if is_relevant:
                    cached_relevant.append(term)

            self._save_cache()

        except Exception as e:
            logger.warning(f"LLM filter failed: {e}. Returning cached terms only.")

        return cached_relevant
