"""Tests for shared data structures."""

from src.loaders.structures import Topic


def test_topic_enum_has_expected_values():
    """Topic enum contains all expected categories."""
    expected = {"performance", "usability", "features", "pricing", "support", "compatibility", "other"}
    actual = {t.value for t in Topic}
    assert actual == expected


def test_topic_enum_string_comparison():
    """Topic enum supports direct string comparison via str base."""
    assert Topic.PERFORMANCE == "performance"
    assert Topic.USABILITY == "usability"
