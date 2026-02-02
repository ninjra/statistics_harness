from pathlib import Path

from statistic_harness.core.pipeline import Pipeline
from statistic_harness.core.utils import now_iso


def test_template_combined_run(tmp_path, monkeypatch):
    appdata = tmp_path / "appdata"
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata))
    pipeline = Pipeline(appdata, Path("plugins"))

    run_id = pipeline.run(
        Path("tests/fixtures/synth_linear.csv"), ["profile_basic"], {}, 0
    )
    run_row = pipeline.storage.fetch_run(run_id)
    assert run_row
    dataset_version_id = run_row["dataset_version_id"]

    template_id = pipeline.storage.create_template(
        name="Linear",
        fields=[{"name": "x1", "dtype": "float"}, {"name": "y", "dtype": "float"}],
        description=None,
        version=None,
        created_at=now_iso(),
    )
    pipeline.run(
        None,
        ["transform_template"],
        {"transform_template": {"template_id": template_id, "mapping": {"x1": "x1", "y": "y"}}},
        0,
        dataset_version_id=dataset_version_id,
    )

    aggregate_id = pipeline.storage.ensure_template_aggregate_dataset(
        template_id, now_iso()
    )
    combined_run = pipeline.run(
        None, ["profile_basic"], {}, 0, dataset_version_id=aggregate_id
    )
    run_dir = appdata / "runs" / combined_run
    assert (run_dir / "report.json").exists()
