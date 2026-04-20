"""FastAPI server for viewing judge reports."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.judge.report import REPORTS_DIR, list_reports, load_report

app = FastAPI(title="Judge Report Viewer")

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """List all reports."""
    reports = []
    for p in list_reports():
        report = load_report(p)
        reports.append({"path": p.name, "timestamp": report.timestamp})
    return templates.TemplateResponse("index.html", {"request": request, "reports": reports})


@app.get("/report/{filename}", response_class=HTMLResponse)
async def view_report(request: Request, filename: str):
    """View a single report."""
    report = load_report(REPORTS_DIR / filename)
    return templates.TemplateResponse("report.html", {"request": request, "report": report})
