import json
from pathlib import Path

import pytest

from statistic_harness.cli import cmd_replay
from statistic_harness.core.pipeline import Pipeline


def test_replay_ok_and_detects_tamper(tmp_path, monkeypatch):
    appdata = tmp_path / "appdata"
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata))
    pipeline = Pipeline(appdata, Path("plugins"))
    run_id = pipeline.run(Path("tests/fixtures/synth_linear.csv"), ["profile_basic"], {}, 123)

    cmd_replay(run_id)

    report_path = appdata / "runs" / run_id / "report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["tampered"] = True
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(SystemExit):
        cmd_replay(run_id)

