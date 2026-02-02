import json
from pathlib import Path

from jsonschema import validate

from statistic_harness.core.pipeline import Pipeline


def test_report_schema(tmp_path, monkeypatch):
    appdata = tmp_path / "appdata"
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata))
    pipeline = Pipeline(appdata, Path("plugins"))
    run_id = pipeline.run(
        Path("tests/fixtures/synth_linear.csv"), ["profile_basic"], {}, 42
    )
    report_path = appdata / "runs" / run_id / "report.json"
    schema = json.loads(Path("docs/report.schema.json").read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate(instance=report, schema=schema)
