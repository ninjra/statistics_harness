from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts.finalize_optimal_4pillars import _run_status


def test_run_status_reads_status_from_db(tmp_path: Path) -> None:
    db = tmp_path / "state.sqlite"
    con = sqlite3.connect(str(db))
    con.execute("create table runs(run_id text primary key, status text)")
    con.execute("insert into runs(run_id,status) values(?,?)", ("run_a", "completed"))
    con.commit()
    con.close()
    assert _run_status(db, "run_a") == "completed"
    assert _run_status(db, "missing_run") is None

