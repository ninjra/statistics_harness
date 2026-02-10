from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from statistic_harness.core.sql_assist import SqlAssist
from statistic_harness.core.storage import Storage


def test_storage_deny_write_prefixes_blocks_insert(tmp_path: Path) -> None:
    db = tmp_path / "state.sqlite"
    con = sqlite3.connect(db)
    try:
        con.execute("CREATE TABLE template_normalized_abc(x INTEGER)")
        con.commit()
    finally:
        con.close()

    storage = Storage(
        db,
        mode="rw",
        initialize=False,
        deny_write_prefixes=["template_normalized_"],
        allow_write_prefixes=[],
    )
    with storage.connection() as conn, pytest.raises(sqlite3.DatabaseError):
        conn.execute("INSERT INTO template_normalized_abc(x) VALUES (1)")


def test_storage_allow_write_prefixes_allows_insert(tmp_path: Path) -> None:
    db = tmp_path / "state.sqlite"
    con = sqlite3.connect(db)
    try:
        con.execute("CREATE TABLE template_normalized_abc(x INTEGER)")
        con.commit()
    finally:
        con.close()

    storage = Storage(
        db,
        mode="rw",
        initialize=False,
        deny_write_prefixes=["template_normalized_"],
        allow_write_prefixes=["template_normalized_"],
    )
    with storage.connection() as conn:
        conn.execute("INSERT INTO template_normalized_abc(x) VALUES (1)")


def test_sql_assist_exec_plugin_requires_prefix(tmp_path: Path) -> None:
    db = tmp_path / "state.sqlite"
    storage = Storage(db, mode="rw", initialize=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    sql = SqlAssist(
        storage=storage,
        run_dir=run_dir,
        plugin_id="demo",
        schema_hash="deadbeef",
        mode="plugin",
        allowed_prefix="plg__demo__",
    )

    # Allowed (prefixed).
    sql.exec_plugin("CREATE TABLE plg__demo__t(x INTEGER)", query_id="create_ok")

    # Denied (unprefixed).
    with pytest.raises(ValueError):
        sql.exec_plugin("CREATE TABLE other(x INTEGER)", query_id="create_bad")


def test_sql_assist_validate_ro_sql_explain_only(tmp_path: Path) -> None:
    db = tmp_path / "state.sqlite"
    storage = Storage(db, mode="rw", initialize=False)
    with storage.connection() as conn:
        conn.execute("CREATE TABLE t(x INTEGER)")
        conn.execute("INSERT INTO t(x) VALUES (1)")

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    sql = SqlAssist(
        storage=storage,
        run_dir=run_dir,
        plugin_id="demo",
        schema_hash="deadbeef",
        mode="ro",
    )

    sql.validate_ro_sql("SELECT x FROM t", query_id="v_ok")
    assert (run_dir / "artifacts" / "demo" / "sql" / "v_ok.manifest.json").exists()

    with pytest.raises(ValueError):
        sql.validate_ro_sql("CREATE TABLE x(y INTEGER)", query_id="v_bad")
