from __future__ import annotations

from pathlib import Path

from statistic_harness.core.pipeline import Pipeline


def test_cross_dataset_pipeline_fail_closed_still_writes_reports(tmp_path, monkeypatch) -> None:
    appdata = tmp_path / "appdata"
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata))
    fixture = Path("tests/fixtures/synth_linear.csv")
    plugin_ids = ["profile_basic", "transform_entity_resolution_map_v1", "report_bundle"]
    settings = {"strict_prerequisites": True, "datasets": {}, "fields": [{"role": "contracts", "field": "vendor_name1"}]}

    pipeline = Pipeline(appdata, Path("plugins"))
    run_id = pipeline.run(fixture, plugin_ids, settings=settings, run_seed=5151)
    run_dir = appdata / "runs" / run_id
    assert (run_dir / "report.json").exists()
    assert (run_dir / "report.md").exists()
    run_row = pipeline.storage.fetch_run(run_id)
    assert run_row is not None
    assert str(run_row.get("status") or "") in {"partial", "completed"}

