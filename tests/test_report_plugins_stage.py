from __future__ import annotations

from pathlib import Path

from statistic_harness.core.pipeline import Pipeline
from statistic_harness.core.plugin_runner import RunnerResponse, run_plugin_subprocess as real_run_plugin_subprocess
from statistic_harness.core.types import PluginResult
import statistic_harness.core.pipeline as pipeline_mod


def test_selected_report_plugins_run_and_report_bundle_runs_last(tmp_path, monkeypatch):
    # Arrange: real ingest to populate DB; stub everything else to keep the test fast and focused
    # on pipeline orchestration.
    appdata = tmp_path / "appdata"
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata))

    executed: list[str] = []

    def fake_run_plugin_subprocess(spec, request, run_dir, cwd):
        executed.append(spec.plugin_id)
        # Ingest + normalization must run for real so the pipeline can enforce the
        # normalized/template layer before executing any downstream plugins.
        if spec.plugin_id in {"ingest_tabular", "transform_normalize_mixed"}:
            return real_run_plugin_subprocess(spec, request, run_dir, cwd)
        # Minimal schema-valid PluginResult.
        result = PluginResult(
            status="ok",
            summary="stubbed",
            metrics={},
            findings=[],
            artifacts=[],
            budget={"row_limit": None, "sampled": False, "time_limit_ms": None, "cpu_limit_ms": None},
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

    monkeypatch.setattr(pipeline_mod, "run_plugin_subprocess", fake_run_plugin_subprocess)

    pipeline = Pipeline(appdata, Path("plugins"))

    # Act: select a non-default report plugin; pipeline should run report plugins selected
    # (and their report dependencies) and still run report_bundle last.
    run_id = pipeline.run(Path("tests/fixtures/synth_linear.csv"), ["report_decision_bundle_v2"], {}, 42)
    run_dir = appdata / "runs" / run_id

    # Assert: selected report plugins executed, and report_bundle ran after them.
    assert (run_dir / "report.json").exists()
    assert (run_dir / "report.md").exists()

    assert "report_slide_kit_emitter_v2" in executed
    assert "report_decision_bundle_v2" in executed
    assert "report_bundle" in executed

    idx_slide = executed.index("report_slide_kit_emitter_v2")
    idx_decision = executed.index("report_decision_bundle_v2")
    idx_bundle = executed.index("report_bundle")

    assert idx_slide < idx_decision
    assert idx_decision < idx_bundle
