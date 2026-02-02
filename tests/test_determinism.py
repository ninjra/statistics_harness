import json
from pathlib import Path

from statistic_harness.core.pipeline import Pipeline


def test_report_determinism(tmp_path, monkeypatch):
    appdata_1 = tmp_path / "appdata_1"
    appdata_2 = tmp_path / "appdata_2"

    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata_1))
    pipeline_1 = Pipeline(appdata_1, Path("plugins"))
    run_id_1 = pipeline_1.run(
        Path("tests/fixtures/synth_linear.csv"), ["profile_basic"], {}, 123
    )
    report_1 = json.loads(
        (appdata_1 / "runs" / run_id_1 / "report.json").read_text(encoding="utf-8")
    )

    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata_2))
    pipeline_2 = Pipeline(appdata_2, Path("plugins"))
    run_id_2 = pipeline_2.run(
        Path("tests/fixtures/synth_linear.csv"), ["profile_basic"], {}, 123
    )
    report_2 = json.loads(
        (appdata_2 / "runs" / run_id_2 / "report.json").read_text(encoding="utf-8")
    )

    assert report_1["input"] == report_2["input"]
    assert report_1["plugins"] == report_2["plugins"]
