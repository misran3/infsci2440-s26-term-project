"""LLM-based query preprocessing to extract keywords from natural language."""

import logging
import os

from pydantic import BaseModel

from src.llm import get_agent
from src.loaders.structures import PreprocessedQuery

logger = logging.getLogger(__name__)

EXTRACT_PROMPT = """You extract search keywords from natural language queries about software product reviews.

Given a user's question or query, extract AND EXPAND the relevant keywords that would help find matching reviews.

Rules:
1. Extract explicit terms from the query
2. Add semantically related terms that reviewers might use (synonyms, related concepts)
3. Focus on software-related interpretations
4. Return 5-15 keywords total
5. Keywords should be single words only, no phrases

Example:
Query: "How many people mentioned the app crashes?"
Keywords: ["app", "crash", "bug", "error", "freeze", "hang", "unstable", "broken", "fails"]

Example:
Query: "What do users think about the pricing?"
Keywords: ["price", "pricing", "cost", "expensive", "cheap", "affordable", "subscription", "value", "money", "worth"]"""


class ExtractedKeywords(BaseModel):
    """LLM response model for extracted keywords."""

    keywords: list[str]


class QueryPreprocessor:
    """Preprocesses natural language queries into keywords for search."""

    def __init__(self) -> None:
        self._enabled = os.getenv("QUERY_PREPROCESSOR_ENABLED", "true").lower() == "true"

    def _looks_like_natural_language(self, query: str) -> bool:
        """Heuristic check if query is natural language vs keywords."""
        query_lower = query.lower().strip()
        words = query_lower.split()

        question_words = (
            "how", "what", "why", "when", "where", "who", "which",
            "is", "are", "do", "does", "can", "could", "would", "should"
        )
        first_word = words[0] if words else ""
        if first_word in question_words:
            return True

        if "?" in query:
            return True

        nl_markers = ("the", "a", "an", "about", "think", "feel", "mentioned", "said")
        if len(words) > 4 and any(w in nl_markers for w in words):
            return True

        return False

    async def preprocess(self, query: str) -> PreprocessedQuery:
        """Preprocess a query, extracting keywords if it's natural language.

        Args:
            query: User's input (either keywords or natural language).

        Returns:
            PreprocessedQuery with original query, extracted keywords, and flag.
        """
        if not self._enabled:
            return PreprocessedQuery(
                original_query=query,
                extracted_keywords=query.lower().split(),
                was_preprocessed=False,
            )

        if not self._looks_like_natural_language(query):
            return PreprocessedQuery(
                original_query=query,
                extracted_keywords=query.lower().split(),
                was_preprocessed=False,
            )

        agent = get_agent(
            output_type=ExtractedKeywords,
            system_prompt=EXTRACT_PROMPT,
            required=False,
        )

        if agent is None:
            logger.warning("LLM not available for query preprocessing. Using raw query.")
            return PreprocessedQuery(
                original_query=query,
                extracted_keywords=query.lower().split(),
                was_preprocessed=False,
            )

        try:
            result = await agent.run(f"Query: {query}")
            keywords = [k.lower() for k in result.output.keywords]

            if not keywords:
                logger.warning("LLM returned empty keywords. Using raw query.")
                return PreprocessedQuery(
                    original_query=query,
                    extracted_keywords=query.lower().split(),
                    was_preprocessed=False,
                )

            logger.info(f"Preprocessed query into {len(keywords)} keywords: {keywords}")
            return PreprocessedQuery(
                original_query=query,
                extracted_keywords=keywords,
                was_preprocessed=True,
            )

        except Exception as e:
            logger.warning(f"Query preprocessing failed: {e}. Using raw query.")
            return PreprocessedQuery(
                original_query=query,
                extracted_keywords=query.lower().split(),
                was_preprocessed=False,
            )
