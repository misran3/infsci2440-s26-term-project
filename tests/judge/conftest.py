"""Pytest plugin for judge system."""

import pytest
from datetime import datetime

from src.judge.models import JudgeReport
from src.judge.report import save_report
from src.llm.provider import LLMProvider, validate_credentials
from src.loaders.loader import load_reviews


def has_llm_credentials():
    """Check if LLM credentials are available.

    Returns:
        True if credentials are valid, False otherwise.
    """
    from src.llm.provider import get_provider
    provider = get_provider()
    valid, _ = validate_credentials(provider)
    return valid


skip_if_no_credentials = pytest.mark.skipif(
    not has_llm_credentials(),
    reason="LLM credentials not available"
)


class JudgeCollector:
    """Collects test results during session."""

    def __init__(self):
        self.tests = []
        self.dataset = "sample_reviews.csv"
        self.review_count = 0

    def add_test(self, test):
        self.tests.append(test)


@pytest.fixture(scope="session")
def judge_collector():
    """Session-scoped collector for judge results."""
    return JudgeCollector()


@pytest.fixture(scope="session")
def reviews():
    """Load sample reviews once per session."""
    return load_reviews(sample=True, limit=100)


@pytest.fixture(scope="session", autouse=True)
def write_report(judge_collector, reviews):
    """Write JSON report after all tests complete."""
    judge_collector.review_count = len(reviews)

    yield

    if judge_collector.tests:
        report = JudgeReport(
            timestamp=datetime.now(),
            dataset=judge_collector.dataset,
            review_count=judge_collector.review_count,
            tests=judge_collector.tests,
        )
        path = save_report(report)
        print(f"\nJudge report saved to: {path}")
