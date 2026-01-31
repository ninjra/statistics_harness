from pathlib import Path

from statistic_harness.core.pipeline import Pipeline
from statistic_harness.core.utils import get_appdata_dir


def test_pipeline_integration(tmp_path, monkeypatch):
    appdata = tmp_path / "appdata"
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata))
    pipeline = Pipeline(appdata, Path("plugins"))
    run_id = pipeline.run(Path("tests/fixtures/synth_linear.csv"), ["profile_basic"], {}, 42)
    run_dir = appdata / "runs" / run_id
    assert (run_dir / "report.json").exists()
    assert (run_dir / "report.md").exists()
