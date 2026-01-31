from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from statistic_harness.core.pipeline import Pipeline
from statistic_harness.core.report import build_report, write_report
from statistic_harness.core.utils import get_appdata_dir, safe_join

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

pipeline = Pipeline(get_appdata_dir(), Path("plugins"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    specs = pipeline.manager.discover()
    return TEMPLATES.TemplateResponse("index.html", {"request": request, "plugins": specs})


@app.get("/plugins", response_class=HTMLResponse)
async def plugins(request: Request) -> HTMLResponse:
    specs = pipeline.manager.discover()
    return TEMPLATES.TemplateResponse("plugins.html", {"request": request, "plugins": specs})


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    upload_id = uuid.uuid4().hex
    upload_dir = get_appdata_dir() / "uploads" / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / file.filename
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")
    target.write_bytes(content)
    return JSONResponse({"upload_id": upload_id, "filename": file.filename})


@app.post("/api/runs")
async def create_run(
    background: BackgroundTasks,
    upload_id: str = Form(...),
    plugins: str = Form(""),
    run_seed: int = Form(0),
) -> JSONResponse:
    upload_dir = get_appdata_dir() / "uploads" / upload_id
    if not upload_dir.exists():
        raise HTTPException(status_code=404, detail="Upload not found")
    files = list(upload_dir.iterdir())
    if not files:
        raise HTTPException(status_code=400, detail="No uploaded file")
    input_file = files[0]
    plugin_ids = [p for p in plugins.split(",") if p]

    run_id = uuid.uuid4().hex

    def run_pipeline() -> None:
        pipeline.run(input_file, plugin_ids, {}, run_seed, run_id=run_id)
        run_dir = get_appdata_dir() / "runs" / run_id
        report = build_report(pipeline.storage, run_id, run_dir, Path("docs/report.schema.json"))
        write_report(report, run_dir)

    background.add_task(run_pipeline)
    return JSONResponse({"status": "queued", "run_id": run_id})


@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_status(request: Request, run_id: str) -> HTMLResponse:
    return TEMPLATES.TemplateResponse("run.html", {"request": request, "run_id": run_id})


@app.get("/api/runs/{run_id}/report.json")
async def get_report_json(run_id: str) -> FileResponse:
    report_path = get_appdata_dir() / "runs" / run_id / "report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(report_path)


@app.get("/api/runs/{run_id}/report.md")
async def get_report_md(run_id: str) -> FileResponse:
    report_path = get_appdata_dir() / "runs" / run_id / "report.md"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(report_path)


@app.get("/api/runs/{run_id}/artifacts/{plugin_id}/{artifact_path:path}")
async def get_artifact(run_id: str, plugin_id: str, artifact_path: str) -> FileResponse:
    base = get_appdata_dir() / "runs" / run_id / "artifacts" / plugin_id
    try:
        resolved = safe_join(base, artifact_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(resolved)
