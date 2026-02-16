from __future__ import annotations

from statistic_harness.core.storage import Storage
from statistic_harness.core.utils import now_iso


def test_fetch_plugin_executions_decodes_bytes_fields(tmp_path) -> None:
    storage = Storage(tmp_path / "state.sqlite")
    run_id = "run_bytes_1"
    ts = now_iso()
    storage.create_run(
        run_id=run_id,
        created_at=ts,
        status="running",
        upload_id="upload_1",
        input_filename="input.csv",
        canonical_path="input.csv",
        settings={},
        error=None,
        run_seed=123,
    )
    execution_id = storage.start_plugin_execution(
        run_id=run_id,
        plugin_id="demo_plugin",
        plugin_version="0.1.0",
        started_at=ts,
        status="running",
    )
    storage.update_plugin_execution(
        execution_id=execution_id,
        completed_at=ts,
        duration_ms=1,
        status="error",
        exit_code=0,
        cpu_user=None,
        cpu_system=None,
        max_rss=None,
        warnings_count=None,
        stdout="",
        stderr=b"\xffbad-bytes",
    )

    rows = storage.fetch_plugin_executions(run_id)
    assert len(rows) == 1
    assert isinstance(rows[0]["stderr"], str)
    assert "bad-bytes" in rows[0]["stderr"]

