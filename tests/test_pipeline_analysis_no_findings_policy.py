from __future__ import annotations

import json
from pathlib import Path

import statistic_harness.core.pipeline as pipeline_mod
from statistic_harness.core.pipeline import Pipeline
from statistic_harness.core.plugin_runner import (
    RunnerResponse,
    run_plugin_subprocess as real_run_plugin_subprocess,
)
from statistic_harness.core.types import PluginResult


def test_analysis_ok_without_findings_gets_diagnostic_finding(tmp_path, monkeypatch) -> None:
    appdata = tmp_path / "appdata"
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata))

    def fake_run_plugin_subprocess(spec, request, run_dir, cwd):
        if spec.plugin_id in {"ingest_tabular", "transform_normalize_mixed", "transform_template"}:
            return real_run_plugin_subprocess(spec, request, run_dir, cwd)
        if spec.plugin_id == "analysis_anova_auto":
            result = PluginResult(
                status="ok",
                summary="ANOVA complete: significant=0",
                metrics={},
                findings=[],
                artifacts=[],
                error=None,
                references=[],
                debug={},
            )
            return RunnerResponse(
                result=result,
                execution={
                    "started_at": request.get("started_at"),
                    "completed_at": request.get("started_at"),
                    "duration_ms": 0,
                    "cpu_user": None,
                    "cpu_system": None,
                    "max_rss": None,
                    "warnings_count": None,
                },
                stdout="",
                stderr="",
                exit_code=0,
            )
        if spec.type == "report" or spec.type == "llm":
            result = PluginResult(
                status="ok",
                summary=f"{spec.plugin_id} stubbed",
                metrics={},
                findings=[{"kind": "stub", "measurement_type": "measured"}],
                artifacts=[],
                error=None,
                references=[],
                debug={},
            )
            return RunnerResponse(
                result=result,
                execution={
                    "started_at": request.get("started_at"),
                    "completed_at": request.get("started_at"),
                    "duration_ms": 0,
                    "cpu_user": None,
                    "cpu_system": None,
                    "max_rss": None,
                    "warnings_count": None,
                },
                stdout="",
                stderr="",
                exit_code=0,
            )
        return real_run_plugin_subprocess(spec, request, run_dir, cwd)

    monkeypatch.setattr(pipeline_mod, "run_plugin_subprocess", fake_run_plugin_subprocess)

    pipeline = Pipeline(appdata, Path("plugins"))
    run_id = pipeline.run(Path("tests/fixtures/synth_linear.csv"), ["analysis_anova_auto"], {}, 7)

    results = pipeline.storage.fetch_plugin_results(run_id)
    analysis_row = next(row for row in results if row["plugin_id"] == "analysis_anova_auto")
    assert analysis_row["status"] == "ok"
    findings = json.loads(analysis_row.get("findings_json") or "[]")
    assert len(findings) >= 1
    assert str(findings[0].get("kind") or "") == "analysis_no_action_diagnostic"
    assert str(findings[0].get("reason_code") or "") == "NO_SIGNIFICANT_EFFECT"

    run_row = pipeline.storage.fetch_run(run_id)
    assert str(run_row.get("status") or "") == "completed"

    manifest = json.loads((appdata / "runs" / run_id / "run_manifest.json").read_text(encoding="utf-8"))
    summary = dict(manifest.get("summary") or {})
    assert int(summary.get("analysis_ok_without_findings_count") or 0) == 0
