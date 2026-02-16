from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.run_loaded_dataset_full import _resolve_structural_roles, _structural_preflight


def _init_structural_tables(db_path: Path) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            """
            CREATE TABLE dataset_columns (
                dataset_version_id TEXT,
                column_id INTEGER,
                original_name TEXT,
                safe_name TEXT,
                role TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE dataset_templates (
                dataset_version_id TEXT,
                template_id INTEGER,
                mapping_json TEXT,
                updated_at TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE template_fields (
                template_id INTEGER,
                field_id INTEGER,
                safe_name TEXT,
                name TEXT
            )
            """
        )
        con.commit()
    finally:
        con.close()


def _seed_structural_fixture(db_path: Path, *, include_process: bool = True) -> str:
    dataset_version_id = "ds_v1"
    con = sqlite3.connect(db_path)
    try:
        columns = [
            (dataset_version_id, 1, "c1", "c1", "id"),
            (dataset_version_id, 2, "c2", "c2", "end_time"),
            (dataset_version_id, 3, "c3", "c3", "id"),
            (dataset_version_id, 4, "c4", "c4", "id"),
        ]
        con.executemany(
            "INSERT INTO dataset_columns(dataset_version_id,column_id,original_name,safe_name,role) VALUES (?,?,?,?,?)",
            columns,
        )

        mapping: dict[str, dict[str, str]] = {
            "START_DT": {"safe_name": "c2"},
            "MASTER_PROCESS_QUEUE_ID": {"safe_name": "c3"},
            "ASSIGNED_MACHINE_ID": {"safe_name": "c4"},
            "USER_ID": {"safe_name": "c5"},
        }
        fields = [
            (4, 1, "c2", "START_DT"),
            (4, 2, "c3", "MASTER_PROCESS_QUEUE_ID"),
            (4, 3, "c4", "ASSIGNED_MACHINE_ID"),
            (4, 4, "c5", "USER_ID"),
        ]
        if include_process:
            mapping["PROCESS_ID"] = {"safe_name": "c1"}
            fields.append((4, 5, "c1", "PROCESS_ID"))

        payload = json.dumps({"mapping": mapping}, sort_keys=True)
        con.execute(
            "INSERT INTO dataset_templates(dataset_version_id,template_id,mapping_json,updated_at) VALUES (?,?,?,?)",
            (dataset_version_id, 4, payload, "2026-02-16T00:00:00+00:00"),
        )
        con.executemany(
            "INSERT INTO template_fields(template_id,field_id,safe_name,name) VALUES (?,?,?,?)",
            fields,
        )
        con.commit()
    finally:
        con.close()
    return dataset_version_id


def test_resolve_structural_roles_prefers_semantic_mapping(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite"
    _init_structural_tables(db_path)
    dataset_version_id = _seed_structural_fixture(db_path, include_process=True)

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        roles = _resolve_structural_roles(con, dataset_version_id)
    finally:
        con.close()

    assert roles["c1"] == "process_name"
    assert roles["c2"] == "start_time"
    assert roles["c3"] == "master_id"
    assert roles["c4"] == "host_id"
    assert roles["c5"] == "user_id"


def test_structural_preflight_blocks_when_required_role_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite"
    _init_structural_tables(db_path)
    dataset_version_id = _seed_structural_fixture(db_path, include_process=False)
    run_dir = tmp_path / "runs" / "r1"

    report = _structural_preflight(
        db_path=db_path,
        dataset_version_id=dataset_version_id,
        plugin_ids=["analysis_param_near_duplicate_minhash_v1"],
        output_dir=run_dir,
    )

    assert report["blocking_count"] == 1
    assert report["blockers"][0]["plugin_id"] == "analysis_param_near_duplicate_minhash_v1"
