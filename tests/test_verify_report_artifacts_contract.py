from __future__ import annotations

from pathlib import Path

from statistic_harness.core.pipeline import Pipeline
from scripts.verify_report_artifacts_contract import verify_report_artifacts


def test_verify_report_artifacts_contract_passes_for_real_run(tmp_path, monkeypatch) -> None:
    appdata = tmp_path / "appdata"
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata))
    pipeline = Pipeline(appdata, Path("plugins"))
    run_id = pipeline.run(Path("tests/fixtures/synth_linear.csv"), ["profile_basic"], {}, 7)
    payload = verify_report_artifacts(appdata / "runs" / run_id)
    assert payload["ok"] is True


def test_verify_report_artifacts_contract_fails_missing_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_x"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = verify_report_artifacts(run_dir)
    assert payload["ok"] is False
    assert "REPORT_JSON_MISSING_OR_EMPTY" in payload["violations"]

