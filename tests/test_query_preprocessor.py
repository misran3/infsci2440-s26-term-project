"""Unit tests for QueryPreprocessor."""

import pytest

from src.search.query_preprocessor import QueryPreprocessor


class TestNaturalLanguageDetection:
    """Tests for _looks_like_natural_language heuristics."""

    def test_question_word_how(self) -> None:
        """Detects 'how' as natural language."""
        qp = QueryPreprocessor()
        assert qp._looks_like_natural_language("How many people mentioned crashes?") is True

    def test_question_word_what(self) -> None:
        """Detects 'what' as natural language."""
        qp = QueryPreprocessor()
        assert qp._looks_like_natural_language("What do users think about pricing?") is True

    def test_question_mark(self) -> None:
        """Detects question mark as natural language."""
        qp = QueryPreprocessor()
        assert qp._looks_like_natural_language("users happy?") is True

    def test_keywords_only(self) -> None:
        """Keywords without NL indicators are not detected."""
        qp = QueryPreprocessor()
        assert qp._looks_like_natural_language("app crash bug") is False

    def test_short_query(self) -> None:
        """Short queries without indicators are keywords."""
        qp = QueryPreprocessor()
        assert qp._looks_like_natural_language("slow performance") is False
