from pathlib import Path
from dataclasses import replace

from statistic_harness.core.pipeline import Pipeline


def _filtered_specs(pipeline: Pipeline, exclude: set[str]) -> list:
    specs = pipeline.manager.discover()
    return [spec for spec in specs if spec.plugin_id not in exclude]


def test_missing_planner_records_error(tmp_path, monkeypatch):
    appdata = tmp_path / "appdata"
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata))
    pipeline = Pipeline(appdata, Path("plugins"))
    filtered = _filtered_specs(pipeline, {"planner_basic"})
    monkeypatch.setattr(pipeline.manager, "discover", lambda: filtered)
    run_id = pipeline.run(
        Path("tests/fixtures/synth_linear.csv"), [], {}, 123
    )
    results = pipeline.storage.fetch_plugin_results(run_id)
    planner = next(
        row for row in results if row["plugin_id"] == "planner_basic"
    )
    assert planner["status"] == "error"


def test_missing_report_still_writes_report(tmp_path, monkeypatch):
    appdata = tmp_path / "appdata"
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata))
    pipeline = Pipeline(appdata, Path("plugins"))
    filtered = _filtered_specs(pipeline, {"report_bundle"})
    monkeypatch.setattr(pipeline.manager, "discover", lambda: filtered)
    run_id = pipeline.run(
        Path("tests/fixtures/synth_linear.csv"), [], {}, 123
    )
    run_dir = appdata / "runs" / run_id
    assert (run_dir / "report.json").exists()
    assert (run_dir / "report.md").exists()
    results = pipeline.storage.fetch_plugin_results(run_id)
    report = next(
        row for row in results if row["plugin_id"] == "report_bundle"
    )
    assert report["status"] == "error"


def test_report_failure_still_writes_report(tmp_path, monkeypatch):
    appdata = tmp_path / "appdata"
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata))
    pipeline = Pipeline(appdata, Path("plugins"))
    specs = pipeline.manager.discover()
    broken_specs = []
    for spec in specs:
        if spec.plugin_id == "report_bundle":
            spec = replace(spec, entrypoint="missing.py:Plugin")
        broken_specs.append(spec)
    monkeypatch.setattr(pipeline.manager, "discover", lambda: broken_specs)
    run_id = pipeline.run(Path("tests/fixtures/synth_linear.csv"), [], {}, 123)
    run_dir = appdata / "runs" / run_id
    assert (run_dir / "report.json").exists()
    assert (run_dir / "report.md").exists()
