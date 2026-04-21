"""Unit tests for TermFilter."""

import json
from pathlib import Path

import pytest

from src.search.term_filter import TermFilter


@pytest.fixture
def temp_cache(tmp_path: Path) -> Path:
    """Create a temporary cache file path."""
    return tmp_path / "test_cache.json"


def test_load_empty_cache(temp_cache: Path) -> None:
    """TermFilter initializes with empty cache when file missing."""
    tf = TermFilter(cache_path=temp_cache, validate_credentials=False)
    assert tf.cache == {}


def test_load_existing_cache(temp_cache: Path) -> None:
    """TermFilter loads existing cache from disk."""
    cache_data = {"glitch": True, "pester": False}
    temp_cache.write_text(json.dumps(cache_data))

    tf = TermFilter(cache_path=temp_cache, validate_credentials=False)
    assert tf.cache == cache_data


def test_save_cache(temp_cache: Path) -> None:
    """TermFilter saves cache to disk."""
    tf = TermFilter(cache_path=temp_cache, validate_credentials=False)
    tf.cache = {"bug": True, "insect": False}
    tf._save_cache()

    loaded = json.loads(temp_cache.read_text())
    assert loaded == {"bug": True, "insect": False}


def test_save_cache_creates_directory(tmp_path: Path) -> None:
    """TermFilter creates parent directory if missing."""
    cache_path = tmp_path / "subdir" / "cache.json"
    tf = TermFilter(cache_path=cache_path, validate_credentials=False)
    tf.cache = {"test": True}
    tf._save_cache()

    assert cache_path.exists()


@pytest.mark.asyncio
async def test_filter_uses_cache_and_skips_llm(temp_cache: Path) -> None:
    """Filter returns cached terms without LLM call when all cached."""
    cache_data = {"glitch": True, "pester": False, "bug": True}
    temp_cache.write_text(json.dumps(cache_data))

    tf = TermFilter(cache_path=temp_cache, validate_credentials=False)
    result = await tf.filter("bug", ["glitch", "pester", "bug"])

    assert set(result) == {"glitch", "bug"}


@pytest.mark.asyncio
async def test_filter_always_includes_query_terms(temp_cache: Path) -> None:
    """Filter always includes original query terms."""
    cache_data = {"glitch": True, "pester": False}
    temp_cache.write_text(json.dumps(cache_data))

    tf = TermFilter(cache_path=temp_cache, validate_credentials=False)
    result = await tf.filter("bug", ["glitch", "pester", "bug"])

    assert "bug" in result


@pytest.mark.asyncio
async def test_filter_calls_llm_for_uncached_terms(temp_cache: Path) -> None:
    """Filter calls LLM for uncached terms and caches results."""
    tf = TermFilter(cache_path=temp_cache, validate_credentials=False)

    result = await tf.filter("bug", ["glitch", "pester", "badger", "bug"])

    assert "bug" in result
    assert "glitch" in result
    assert "pester" not in result
    assert "badger" not in result

    assert "glitch" in tf.cache
    assert tf.cache["glitch"] is True
    assert "pester" in tf.cache
    assert tf.cache["pester"] is False
