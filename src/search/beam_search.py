"""Beam search query expansion using WordNet."""

from __future__ import annotations

from typing import Any

import nltk
from nltk.corpus import wordnet

from src.loaders.structures import QueryExpansion

# Download WordNet if not present
try:
    wordnet.synsets("test")
except LookupError:
    nltk.download("wordnet", quiet=True)


class BeamSearchExpander:
    """Expands queries using beam search over WordNet synonyms."""

    def __init__(self, beam_width: int = 3, max_depth: int = 2) -> None:
        self.beam_width = beam_width
        self.max_depth = max_depth

    def get_synonyms(self, word: str) -> list[str]:
        """Get top 5 synonyms from WordNet, excluding underscored lemmas."""
        synonyms: set[str] = set()
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                name = lemma.name().lower()
                if name != word.lower() and "_" not in name:
                    synonyms.add(name)
        return list(synonyms)[:5]

    def score_expansion(self, original: str, expanded: str) -> float:
        """Score expansion by diversity while maintaining relevance.

        Rewards adding new synonym terms (diversity) while keeping
        some connection to the original query (relevance). This
        encourages meaningful query expansion rather than penalizing it.
        """
        original_words = set(original.lower().split())
        expanded_words = set(expanded.lower().split())

        # New terms added (encourage expansion)
        new_terms = expanded_words - original_words
        # Terms retained from original (maintain relevance)
        retained = original_words & expanded_words

        diversity_score = len(new_terms) * 0.4
        relevance_score = len(retained) / max(len(original_words), 1) * 0.5

        return diversity_score + relevance_score + 0.1

    def expand(self, query: str) -> QueryExpansion:
        """Expand query using beam search over synonyms."""
        if not query.strip():
            return QueryExpansion(
                original_query=query,
                expanded_terms=[],
                beam_paths=[],
            )

        words = query.lower().split()
        beam_paths: list[dict[str, Any]] = []

        # Current beam: list of (expansion_string, score, path)
        beam: list[tuple[str, float, list[str]]] = [(query.lower(), 1.0, [query.lower()])]

        for depth in range(self.max_depth):
            candidates: list[tuple[str, float, list[str]]] = []

            for current, score, path in beam:
                current_words = current.split()

                for i, word in enumerate(current_words):
                    synonyms = self.get_synonyms(word)

                    for syn in synonyms:
                        new_words = current_words.copy()
                        new_words[i] = syn
                        new_expansion = " ".join(new_words)

                        new_score = self.score_expansion(query, new_expansion)
                        new_path = path + [new_expansion]

                        candidates.append((new_expansion, new_score, new_path))

            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[: self.beam_width]

            # Record paths for visualization
            for exp, sc, p in beam:
                beam_paths.append({"path": p, "score": sc})

        # Extract terms only from final beam winners (proper beam search behavior)
        final_terms: set[str] = set(words)
        for expansion, _, _ in beam:
            final_terms.update(expansion.split())

        return QueryExpansion(
            original_query=query,
            expanded_terms=list(final_terms),
            beam_paths=beam_paths[:10],
        )
