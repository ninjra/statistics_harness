from __future__ import annotations

import importlib.util
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _load_module():
    path = Path("scripts/repair_stale_running_runs.py").resolve()
    spec = importlib.util.spec_from_file_location("repair_stale_running_runs", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _init_db(db_path: Path) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT,
                completed_at TEXT,
                error_json TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE plugin_executions (
                run_id TEXT NOT NULL,
                status TEXT NOT NULL,
                completed_at TEXT,
                duration_ms INTEGER,
                stderr TEXT
            )
            """
        )
        con.commit()
    finally:
        con.close()


def test_repair_stale_runs_skips_fresh_run_without_journal(tmp_path: Path) -> None:
    module = _load_module()
    appdata = tmp_path / "appdata"
    runs_dir = appdata / "runs"
    runs_dir.mkdir(parents=True)
    db_path = appdata / "state.sqlite"
    _init_db(db_path)

    now = datetime.now(timezone.utc).isoformat()
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            "INSERT INTO runs(run_id, status, created_at) VALUES(?, 'running', ?)",
            ("run_fresh", now),
        )
        con.commit()
    finally:
        con.close()

    result = module.repair_stale_runs(appdata, startup_grace_seconds=180)
    assert result.repaired == 0

    con = sqlite3.connect(db_path)
    try:
        status = con.execute("SELECT status FROM runs WHERE run_id='run_fresh'").fetchone()[0]
    finally:
        con.close()
    assert status == "running"


def test_repair_stale_runs_repairs_old_run_without_journal(tmp_path: Path) -> None:
    module = _load_module()
    appdata = tmp_path / "appdata"
    runs_dir = appdata / "runs"
    runs_dir.mkdir(parents=True)
    db_path = appdata / "state.sqlite"
    _init_db(db_path)

    old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            "INSERT INTO runs(run_id, status, created_at) VALUES(?, 'running', ?)",
            ("run_old", old),
        )
        con.execute(
            "INSERT INTO plugin_executions(run_id, status) VALUES(?, 'running')",
            ("run_old",),
        )
        con.commit()
    finally:
        con.close()

    result = module.repair_stale_runs(appdata, startup_grace_seconds=60)
    assert result.repaired == 1
    assert result.repaired_run_ids == ["run_old"]

    con = sqlite3.connect(db_path)
    try:
        run_status = con.execute("SELECT status FROM runs WHERE run_id='run_old'").fetchone()[0]
        plugin_status = con.execute(
            "SELECT status FROM plugin_executions WHERE run_id='run_old'"
        ).fetchone()[0]
    finally:
        con.close()
    assert run_status == "aborted"
    assert plugin_status == "aborted"
