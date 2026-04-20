"""LLM-based term filtering for search expansion."""

import json
import logging
from pathlib import Path

from pydantic import BaseModel

from src.config import DATA_DIR

logger = logging.getLogger(__name__)


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
