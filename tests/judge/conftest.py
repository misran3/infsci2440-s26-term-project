"""Pytest plugin for judge system."""

import pytest
from datetime import datetime

from src.judge.models import JudgeReport
from src.judge.report import save_report
from src.loaders.loader import load_reviews


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
