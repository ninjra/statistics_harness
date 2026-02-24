from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import scripts.verify_plugin_result_contract as contract_mod


def _init_db(db_path: Path) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.executescript(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                status TEXT,
                dataset_version_id TEXT,
                created_at TEXT,
                completed_at TEXT
            );
            CREATE TABLE plugin_executions (
                execution_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                plugin_id TEXT,
                status TEXT,
                exit_code INTEGER
            );
            CREATE TABLE plugin_results_v2 (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                plugin_id TEXT,
                status TEXT,
                summary TEXT,
                findings_json TEXT,
                error_json TEXT
            );
            """
        )
        con.commit()
    finally:
        con.close()


def test_plugin_result_contract_passes_for_ok_and_na(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "state.sqlite"
    _init_db(db_path)
    run_id = "r1"
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            "INSERT INTO runs(run_id,status,dataset_version_id,created_at,completed_at) VALUES (?,?,?,?,?)",
            (run_id, "completed", "ds1", "2026-02-24T00:00:00+00:00", "2026-02-24T00:05:00+00:00"),
        )
        con.execute(
            "INSERT INTO plugin_executions(run_id,plugin_id,status,exit_code) VALUES (?,?,?,?)",
            (run_id, "analysis_ok", "ok", 0),
        )
        con.execute(
            "INSERT INTO plugin_executions(run_id,plugin_id,status,exit_code) VALUES (?,?,?,?)",
            (run_id, "analysis_na", "na", 0),
        )
        con.execute(
            "INSERT INTO plugin_results_v2(run_id,plugin_id,status,summary,findings_json,error_json) VALUES (?,?,?,?,?,?)",
            (run_id, "analysis_ok", "ok", "done", "[]", "{}"),
        )
        con.execute(
            "INSERT INTO plugin_results_v2(run_id,plugin_id,status,summary,findings_json,error_json) VALUES (?,?,?,?,?,?)",
            (
                run_id,
                "analysis_na",
                "na",
                "not applicable",
                json.dumps([{"kind": "plugin_not_applicable", "reason_code": "PREREQUISITE_UNMET"}]),
                "{}",
            ),
        )
        con.commit()
    finally:
        con.close()

    run_dir = tmp_path / "appdata" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "plugins": {
                    "analysis_ok": {"status": "ok", "findings": []},
                    "analysis_na": {"status": "na", "findings": [{"kind": "plugin_not_applicable"}]},
                },
                "recommendations": {
                    "explanations": {
                        "items": [{"plugin_id": "analysis_na", "reason_code": "PREREQUISITE_UNMET"}]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(contract_mod, "ROOT", tmp_path)
    payload = contract_mod.audit_plugin_contract(db_path, run_id)
    assert payload["ok"] is True
    assert int(payload["violation_count"] or 0) == 0


def test_plugin_result_contract_flags_missing_report_snapshot(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "state.sqlite"
    _init_db(db_path)
    run_id = "r2"
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            "INSERT INTO runs(run_id,status,dataset_version_id,created_at,completed_at) VALUES (?,?,?,?,?)",
            (run_id, "completed", "ds1", "2026-02-24T00:00:00+00:00", "2026-02-24T00:05:00+00:00"),
        )
        con.execute(
            "INSERT INTO plugin_executions(run_id,plugin_id,status,exit_code) VALUES (?,?,?,?)",
            (run_id, "analysis_missing_snapshot", "ok", 0),
        )
        con.execute(
            "INSERT INTO plugin_results_v2(run_id,plugin_id,status,summary,findings_json,error_json) VALUES (?,?,?,?,?,?)",
            (run_id, "analysis_missing_snapshot", "ok", "done", "[]", "{}"),
        )
        con.commit()
    finally:
        con.close()

    run_dir = tmp_path / "appdata" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(
        json.dumps({"plugins": {}, "recommendations": {"explanations": {"items": []}}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(contract_mod, "ROOT", tmp_path)
    payload = contract_mod.audit_plugin_contract(db_path, run_id)
    assert payload["ok"] is False
    assert int(payload["violation_count"] or 0) == 1
