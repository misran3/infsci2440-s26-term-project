"""Unit tests for QueryPreprocessor."""

import pytest
from unittest.mock import patch

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


class TestPreprocessKeywordPassthrough:
    """Tests for keyword passthrough when not NL."""

    @pytest.mark.asyncio
    async def test_keywords_pass_through_unchanged(self) -> None:
        """Keywords are split and returned without LLM call."""
        qp = QueryPreprocessor()
        result = await qp.preprocess("app crash bug")

        assert result.original_query == "app crash bug"
        assert set(result.extracted_keywords) == {"app", "crash", "bug"}
        assert result.was_preprocessed is False

    @pytest.mark.asyncio
    async def test_disabled_returns_split_query(self) -> None:
        """When disabled, always returns split query."""
        with patch.dict("os.environ", {"QUERY_PREPROCESSOR_ENABLED": "false"}):
            qp = QueryPreprocessor()
            result = await qp.preprocess("How many crashes?")

            assert result.was_preprocessed is False
            assert "how" in result.extracted_keywords


class TestPreprocessFallback:
    """Tests for fallback when LLM unavailable."""

    @pytest.fixture
    def mock_no_llm(self):
        """Mock LLM as unavailable."""
        with patch("src.search.query_preprocessor.get_agent", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_fallback_when_llm_unavailable(self, mock_no_llm) -> None:
        """Falls back to split query when LLM unavailable."""
        qp = QueryPreprocessor()
        result = await qp.preprocess("How many people mentioned crashes?")

        assert result.was_preprocessed is False
        assert "crashes" in result.extracted_keywords or "crashes?" in result.extracted_keywords
