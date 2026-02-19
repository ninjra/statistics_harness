from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts.run_run_status import _get_run_status


def test_plugin_done_counts_na_terminal_status(tmp_path: Path) -> None:
    appdata = tmp_path / "appdata"
    appdata.mkdir(parents=True, exist_ok=True)
    db_path = appdata / "state.sqlite"
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                status TEXT,
                created_at TEXT,
                input_filename TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE plugin_executions (
                run_id TEXT,
                plugin_id TEXT,
                status TEXT,
                started_at TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE events (
                run_id TEXT,
                kind TEXT,
                created_at TEXT,
                payload_json TEXT
            )
            """
        )
        con.execute(
            "INSERT INTO runs(run_id, status, created_at, input_filename) VALUES (?, ?, ?, ?)",
            ("run_1", "running", "2026-02-19T00:00:00+00:00", "db://sample"),
        )
        con.executemany(
            "INSERT INTO plugin_executions(run_id, plugin_id, status, started_at) VALUES (?, ?, ?, ?)",
            [
                ("run_1", "p_ok", "ok", None),
                ("run_1", "p_na", "na", None),
                ("run_1", "p_running", "running", "2026-02-19T00:01:00+00:00"),
            ],
        )
        con.commit()
    finally:
        con.close()

    status = _get_run_status(appdata, "run_1")
    assert status.plugin_done == 2
    assert status.plugin_running in {0, 1}
