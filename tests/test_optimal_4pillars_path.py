from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from scripts import optimal_4pillars_path as mod


def _seed_db(path: Path) -> None:
    con = sqlite3.connect(str(path))
    con.executescript(
        """
        CREATE TABLE plugin_executions (
            execution_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            plugin_id TEXT,
            plugin_version TEXT,
            started_at TEXT,
            completed_at TEXT,
            duration_ms INTEGER,
            status TEXT,
            exit_code INTEGER,
            cpu_user REAL,
            cpu_system REAL,
            max_rss INTEGER,
            warnings_count INTEGER,
            stdout TEXT,
            stderr TEXT,
            tenant_id TEXT
        );
        """
    )
    con.executemany(
        """
        INSERT INTO plugin_executions (
            run_id, plugin_id, plugin_version, started_at, completed_at, duration_ms,
            status, exit_code, cpu_user, cpu_system, max_rss, warnings_count, stdout, stderr, tenant_id
        ) VALUES (?, ?, '0.1.0', '', '', ?, ?, ?, 0, 0, ?, 0, '', ?, '')
        """,
        [
            ("run_1", "plugin_ok", 1000, "ok", 0, 1200, ""),
            ("run_1", "plugin_mem", 5000, "error", 0, 2200000, "MemoryError: unable to allocate"),
            ("run_1", "plugin_kill", 600000, "error", -9, 1000, ""),
        ],
    )
    con.commit()
    con.close()


def test_optimal_4pillars_path_generates_triage(tmp_path: Path) -> None:
    db = tmp_path / "state.sqlite"
    out_dir = tmp_path / "out"
    _seed_db(db)
    script = Path(mod.__file__).resolve()
    cp = subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-id",
            "run_1",
            "--db",
            str(db),
            "--out-dir",
            str(out_dir),
            "--top-n",
            "5",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0, cp.stderr

    out_json = out_dir / "optimal_4pillars_triage_run_1.json"
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["totals"]["plugins"] == 3
    assert payload["totals"]["error"] == 2
    assert payload["priority_queue"][0]["plugin_id"] in {"plugin_mem", "plugin_kill"}
