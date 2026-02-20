from __future__ import annotations

from pathlib import Path

import pytest

import statistic_harness.core.pipeline as pipeline_mod
from statistic_harness.core.pipeline import Pipeline
from statistic_harness.core.plugin_runner import (
    RunnerResponse,
    run_plugin_subprocess as real_run_plugin_subprocess,
)
from statistic_harness.core.types import PluginResult


@pytest.mark.parametrize("source_status", ["skipped", "degraded"])
def test_pipeline_normalizes_legacy_nonterminal_statuses_to_ok_observation(
    tmp_path, monkeypatch, source_status: str
) -> None:
    appdata = tmp_path / "appdata"
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata))

    def fake_run_plugin_subprocess(spec, request, run_dir, cwd):
        if spec.plugin_id in {"ingest_tabular", "transform_normalize_mixed"}:
            return real_run_plugin_subprocess(spec, request, run_dir, cwd)
        if spec.plugin_id == "profile_basic":
            result = PluginResult(
                status=source_status,
                summary="preconditions not met",
                metrics={},
                findings=[],
                artifacts=[],
                budget={
                    "row_limit": None,
                    "sampled": False,
                    "time_limit_ms": None,
                    "cpu_limit_ms": None,
                },
                error=None,
                references=[],
                debug={},
            )
        else:
            return real_run_plugin_subprocess(spec, request, run_dir, cwd)
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

    monkeypatch.setattr(pipeline_mod, "run_plugin_subprocess", fake_run_plugin_subprocess)

    pipeline = Pipeline(appdata, Path("plugins"))
    run_id = pipeline.run(Path("tests/fixtures/synth_linear.csv"), ["profile_basic"], {}, 123)

    results = pipeline.storage.fetch_plugin_results(run_id)
    profile_result = next(row for row in results if row["plugin_id"] == "profile_basic")
    assert profile_result["status"] == "ok"
    assert "observation" in str(profile_result["summary"] or "").lower()

    executions = pipeline.storage.fetch_plugin_executions(run_id)
    profile_exec = next(row for row in executions if row["plugin_id"] == "profile_basic")
    assert profile_exec["status"] == "ok"

    run_row = pipeline.storage.fetch_run(run_id)
    assert str(run_row.get("status") or "") == "completed"
