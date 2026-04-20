"""Judge system for semantic evaluation of search expansion."""

from src.judge.heuristics import (
    check_expansion_not_empty,
    check_recall_improvement,
    check_score_ordering,
)
from src.judge.llm_judge import evaluate_result_relevance, evaluate_term_relevance
from src.judge.models import (
    HeuristicTest,
    JudgeReport,
    ResultJudgment,
    ResultRelevanceBatchResult,
    ResultRelevanceTest,
    TermJudgment,
    TermRelevanceBatchResult,
    TermRelevanceTest,
)
from src.judge.report import list_reports, load_all_reports, load_report, save_report

__all__ = [
    "check_expansion_not_empty",
    "check_recall_improvement",
    "check_score_ordering",
    "evaluate_result_relevance",
    "evaluate_term_relevance",
    "HeuristicTest",
    "JudgeReport",
    "list_reports",
    "load_all_reports",
    "load_report",
    "ResultJudgment",
    "ResultRelevanceBatchResult",
    "ResultRelevanceTest",
    "save_report",
    "TermJudgment",
    "TermRelevanceBatchResult",
    "TermRelevanceTest",
]
