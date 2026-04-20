"""Tests for configuration module."""

from pathlib import Path


def test_data_config_paths_are_pathlib():
    """Config paths should be Path objects."""
    from src.config import DATA

    assert isinstance(DATA.raw_reviews, Path)
    assert isinstance(DATA.main_corpus, Path)
    assert isinstance(DATA.sample_reviews, Path)
    assert isinstance(DATA.labeled_reviews, Path)
    assert isinstance(DATA.curated_labels, Path)


def test_data_config_paths_in_data_dir():
    """All data paths should be under data/ directory."""
    from src.config import DATA, DATA_DIR

    assert DATA.raw_reviews.parent == DATA_DIR
    assert DATA.main_corpus.parent == DATA_DIR


def test_labeling_config_has_all_topics():
    """Labeling config should have all expected topics."""
    from src.config import LABELING

    expected_topics = {"performance", "usability", "features", "pricing", "support", "compatibility", "other"}
    assert set(LABELING.topics) == expected_topics


def test_labeling_config_keywords_not_empty():
    """Each topic should have keywords defined."""
    from src.config import LABELING

    for topic in LABELING.topics:
        if topic != "other":
            assert topic in LABELING.topic_keywords
            assert len(LABELING.topic_keywords[topic]) > 0
