from __future__ import annotations

from pathlib import Path

from statistic_harness.core.pipeline import Pipeline


def test_report_outputs_exist_and_are_non_empty(tmp_path, monkeypatch):
    appdata = tmp_path / "appdata"
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata))
    pipeline = Pipeline(appdata, Path("plugins"))
    run_id = pipeline.run(Path("tests/fixtures/synth_linear.csv"), ["profile_basic"], {}, 42)
    run_dir = appdata / "runs" / run_id

    report_json = run_dir / "report.json"
    report_md = run_dir / "report.md"

    assert report_json.exists()
    assert report_json.stat().st_size > 0
    assert report_md.exists()
    assert report_md.stat().st_size > 0

