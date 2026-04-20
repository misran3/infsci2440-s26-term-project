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
    tf = TermFilter(cache_path=temp_cache)
    assert tf.cache == {}


def test_load_existing_cache(temp_cache: Path) -> None:
    """TermFilter loads existing cache from disk."""
    cache_data = {"glitch": True, "pester": False}
    temp_cache.write_text(json.dumps(cache_data))

    tf = TermFilter(cache_path=temp_cache)
    assert tf.cache == cache_data


def test_save_cache(temp_cache: Path) -> None:
    """TermFilter saves cache to disk."""
    tf = TermFilter(cache_path=temp_cache)
    tf.cache = {"bug": True, "insect": False}
    tf._save_cache()

    loaded = json.loads(temp_cache.read_text())
    assert loaded == {"bug": True, "insect": False}


def test_save_cache_creates_directory(tmp_path: Path) -> None:
    """TermFilter creates parent directory if missing."""
    cache_path = tmp_path / "subdir" / "cache.json"
    tf = TermFilter(cache_path=cache_path)
    tf.cache = {"test": True}
    tf._save_cache()

    assert cache_path.exists()
