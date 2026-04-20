"""Pydantic models for judge system."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class TermJudgment(BaseModel):
    """LLM judgment for a single expanded term."""

    term: str
    is_relevant: bool
    reasoning: str


class TermRelevanceBatchResult(BaseModel):
    """Batch result for term relevance evaluation."""

    query: str
    judgments: list[TermJudgment]


class ResultJudgment(BaseModel):
    """LLM judgment for a single retrieved review."""

    review_id: str
    is_relevant: bool
    reasoning: str


class ResultRelevanceBatchResult(BaseModel):
    """Batch result for result relevance evaluation."""

    query: str
    judgments: list[ResultJudgment]


class TermRelevanceTest(BaseModel):
    """Term relevance test result for report."""

    name: str = "term_relevance"
    query: str
    expanded_terms: list[str]
    judgments: list[TermJudgment]


class ResultRelevanceTest(BaseModel):
    """Result relevance test result for report."""

    name: str = "result_relevance"
    query: str
    results: list[ResultJudgment]


class HeuristicTest(BaseModel):
    """Heuristic test result for report."""

    name: str
    query: str
    passed: bool
    details: dict[str, Any]


class JudgeReport(BaseModel):
    """Complete judge report."""

    timestamp: datetime
    dataset: str
    review_count: int
    tests: list[TermRelevanceTest | ResultRelevanceTest | HeuristicTest]
