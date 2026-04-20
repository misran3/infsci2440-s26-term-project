"""JSON report save/load utilities."""

from pathlib import Path

from src.config import DATA_DIR
from src.judge.models import JudgeReport

REPORTS_DIR = DATA_DIR / "judge_reports"


def save_report(report: JudgeReport) -> Path:
    """Save report to JSON file."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"report_{report.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    path = REPORTS_DIR / filename

    path.write_text(report.model_dump_json(indent=2))
    return path


def load_report(path: Path) -> JudgeReport:
    """Load report from JSON file."""
    return JudgeReport.model_validate_json(path.read_text())


def list_reports() -> list[Path]:
    """List all report files, newest first."""
    if not REPORTS_DIR.exists():
        return []
    return sorted(REPORTS_DIR.glob("report_*.json"), reverse=True)


def load_all_reports() -> list[JudgeReport]:
    """Load all reports, newest first."""
    return [load_report(p) for p in list_reports()]
