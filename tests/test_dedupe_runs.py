from pathlib import Path

import pandas as pd

from statistic_harness.core.pipeline import Pipeline


def test_dedupe_by_content(tmp_path, monkeypatch):
    appdata = tmp_path / "appdata"
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata))
    pipeline = Pipeline(appdata, Path("plugins"))

    run_id_1 = pipeline.run(
        Path("tests/fixtures/synth_linear.csv"), ["profile_basic"], {}, 42
    )
    run_id_2 = pipeline.run(
        Path("tests/fixtures/synth_linear.csv"), ["profile_basic"], {}, 42
    )

    run_1 = pipeline.storage.fetch_run(run_id_1)
    run_2 = pipeline.storage.fetch_run(run_id_2)

    assert run_1
    assert run_2
    assert run_1["project_id"] == run_2["project_id"]
    assert run_1["dataset_id"] == run_2["dataset_id"]
    assert run_1["dataset_version_id"] == run_2["dataset_version_id"]

    other_path = tmp_path / "other.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 5]}).to_csv(other_path, index=False)
    run_id_3 = pipeline.run(other_path, ["profile_basic"], {}, 42)
    run_3 = pipeline.storage.fetch_run(run_id_3)
    assert run_3
    assert run_3["dataset_id"] != run_1["dataset_id"]
