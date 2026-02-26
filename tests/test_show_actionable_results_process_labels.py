from __future__ import annotations

import sqlite3
from pathlib import Path

import scripts.show_actionable_results as sar


def _seed_state_db(appdata: Path) -> None:
    appdata.mkdir(parents=True, exist_ok=True)
    db_path = appdata / "state.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE dataset_versions (
                dataset_version_id TEXT PRIMARY KEY,
                table_name TEXT
            );
            CREATE TABLE dataset_templates (
                dataset_version_id TEXT,
                template_id INTEGER,
                updated_at TEXT
            );
            CREATE TABLE template_fields (
                template_id INTEGER,
                field_id INTEGER,
                name TEXT
            );
            CREATE TABLE dv_rows (
                row_id INTEGER,
                row_index INTEGER,
                c1 INTEGER,
                c2 TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO dataset_versions(dataset_version_id, table_name) VALUES(?, ?)",
            ("dv1", "dv_rows"),
        )
        conn.execute(
            "INSERT INTO dataset_templates(dataset_version_id, template_id, updated_at) VALUES(?, ?, ?)",
            ("dv1", 7, "2026-02-26T00:00:00Z"),
        )
        conn.executemany(
            "INSERT INTO template_fields(template_id, field_id, name) VALUES(?, ?, ?)",
            [
                (7, 1, "PROCESS_QUEUE_ID"),
                (7, 2, "PROCESS_ID"),
            ],
        )
        conn.execute(
            "INSERT INTO dv_rows(row_id, row_index, c1, c2) VALUES(?, ?, ?, ?)",
            (295630, 295630, 10187033, "LOSEXTCHLD"),
        )
        conn.commit()
    finally:
        conn.close()


def test_display_process_label_resolves_numeric_id_from_evidence(monkeypatch, tmp_path: Path) -> None:
    appdata = tmp_path / "appdata"
    _seed_state_db(appdata)
    monkeypatch.setattr(sar, "APPDATA", appdata)
    report = {"input": {"filename": "db://dv1"}}
    resolver = sar._build_numeric_process_resolver(report)
    item = {"evidence": [{"row_ids": [295630]}]}

    value = sar._display_process_label("10187033", item, resolver)
    assert value == "LOSEXTCHLD [10187033]"


def test_display_process_label_returns_deterministic_unknown_when_unresolved() -> None:
    value = sar._display_process_label("99999999", {"evidence": []}, resolver=None)
    assert value == "unknown_process(id=99999999)"


def test_display_process_label_keeps_non_numeric_process_id() -> None:
    value = sar._display_process_label("RPT_POR002", {"evidence": []}, resolver=None)
    assert value == "RPT_POR002"


def test_humanize_recommendation_process_rewrites_numeric_process_token() -> None:
    text = "Process 10187033 is a top wait contributor; tune retries for `10187033`."
    out = sar._humanize_recommendation_process(text, "10187033", "LOSEXTCHLD [10187033]")
    assert "Process LOSEXTCHLD [10187033] is a top wait contributor" in out
    assert "`LOSEXTCHLD [10187033]`" in out
