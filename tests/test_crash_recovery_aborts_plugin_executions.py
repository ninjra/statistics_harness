from __future__ import annotations

import os
from pathlib import Path

from statistic_harness.core.pipeline import Pipeline
from statistic_harness.core.storage import Storage
from statistic_harness.core.utils import now_iso, write_json


def test_crash_recovery_marks_run_and_plugin_executions_aborted(tmp_path: Path, monkeypatch) -> None:
    # Arrange: create a run with a dead pid and a running plugin execution.
    appdata = tmp_path / "appdata"
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata))

    storage = Storage(appdata / "state.sqlite")
    run_id = "deadbeefdeadbeefdeadbeefdeadbeef"
    run_dir = appdata / "runs" / run_id
    (run_dir / "dataset").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "journal.json",
        {
            "run_id": run_id,
            "status": "running",
            # PID 999999 should not exist.
            "pid": 999999,
            "started_at": now_iso(),
        },
    )
    storage.create_run(
        run_id=run_id,
        created_at=now_iso(),
        status="running",
        upload_id="local",
        input_filename="db://fake",
        canonical_path=str(run_dir / "dataset" / "canonical.csv"),
        settings={},
        error=None,
        run_seed=1,
        requested_run_seed=1,
        project_id="p",
        dataset_id="d",
        dataset_version_id="dv",
        input_hash="h",
    )
    storage.start_plugin_execution(
        run_id=run_id,
        plugin_id="analysis_dummy",
        plugin_version="0.1.0",
        started_at=now_iso(),
        status="running",
    )

    # Act: Pipeline init triggers crash recovery.
    Pipeline(appdata, Path("plugins"))

    # Assert: run is aborted and plugin execution is no longer running.
    run = storage.fetch_run(run_id)
    assert run is not None
    assert run.get("status") == "aborted"
    executions = storage.fetch_plugin_executions(run_id)
    assert executions
    assert all(e.get("status") != "running" for e in executions)
