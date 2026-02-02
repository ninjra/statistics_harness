from __future__ import annotations

from pathlib import Path
import json
import sqlite3

from statistic_harness.core.migrations import MIGRATIONS
from statistic_harness.core.utils import DEFAULT_TENANT_ID, now_iso, json_dumps


FIXTURE_DIR = Path(__file__).resolve().parent


def apply_migrations(conn: sqlite3.Connection, up_to: int) -> None:
    for version, migration in enumerate(MIGRATIONS[:up_to], start=1):
        migration(conn)
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            (version, now_iso()),
        )
        conn.execute(f"PRAGMA user_version = {version}")
    conn.commit()


def seed_base(conn: sqlite3.Connection, version: int) -> None:
    run_id = f"run_v{version}"
    if version >= 16:
        conn.execute(
            "INSERT INTO runs (run_id, tenant_id, created_at, status, upload_id, input_filename, canonical_path, settings_json, error_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                DEFAULT_TENANT_ID,
                now_iso(),
                "completed",
                "upload",
                "file.csv",
                "path.csv",
                "{}",
                None,
            ),
        )
    else:
        conn.execute(
            "INSERT INTO runs (run_id, created_at, status, upload_id, input_filename, canonical_path, settings_json, error_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                now_iso(),
                "completed",
                "upload",
                "file.csv",
                "path.csv",
                "{}",
                None,
            ),
        )
    conn.execute(
        "INSERT INTO plugin_results (run_id, plugin_id, status, summary, metrics_json, findings_json, artifacts_json, error_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, "analysis_dp_gmm", "ok", "seed", "{}", "[]", "[]", None),
    )

    if version >= 2:
        if version >= 16:
            conn.execute(
                "INSERT INTO projects (project_id, tenant_id, fingerprint, name, created_at) VALUES (?, ?, ?, ?, ?)",
                ("project_seed", DEFAULT_TENANT_ID, "fp_seed", "Project Seed", now_iso()),
            )
            conn.execute(
                "INSERT INTO datasets (dataset_id, tenant_id, project_id, fingerprint, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    "dataset_seed",
                    DEFAULT_TENANT_ID,
                    "project_seed",
                    "fp_seed",
                    now_iso(),
                ),
            )
            conn.execute(
                "INSERT INTO dataset_versions (dataset_version_id, tenant_id, dataset_id, created_at, table_name, data_hash, row_count, column_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "dv_seed",
                    DEFAULT_TENANT_ID,
                    "dataset_seed",
                    now_iso(),
                    "dataset_seed_table",
                    "hash",
                    1,
                    1,
                ),
            )
            conn.execute(
                "INSERT INTO dataset_columns (dataset_version_id, tenant_id, column_id, safe_name, original_name, dtype, role) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("dv_seed", DEFAULT_TENANT_ID, 1, "c1", "col1", "float", None),
            )
        else:
            conn.execute(
                "INSERT INTO projects (project_id, fingerprint, name, created_at) VALUES (?, ?, ?, ?)",
                ("project_seed", "fp_seed", "Project Seed", now_iso()),
            )
            conn.execute(
                "INSERT INTO datasets (dataset_id, project_id, fingerprint, created_at) VALUES (?, ?, ?, ?)",
                ("dataset_seed", "project_seed", "fp_seed", now_iso()),
            )
            conn.execute(
                "INSERT INTO dataset_versions (dataset_version_id, dataset_id, created_at, table_name, data_hash, row_count, column_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("dv_seed", "dataset_seed", now_iso(), "dataset_seed_table", "hash", 1, 1),
            )
            conn.execute(
                "INSERT INTO dataset_columns (dataset_version_id, column_id, safe_name, original_name, dtype, role) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("dv_seed", 1, "c1", "col1", "float", None),
            )

    if version >= 3:
        if version >= 16:
            conn.execute(
                "INSERT INTO plugin_results_v2 (run_id, tenant_id, plugin_id, plugin_version, executed_at, code_hash, settings_hash, dataset_hash, status, summary, metrics_json, findings_json, artifacts_json, error_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    DEFAULT_TENANT_ID,
                    "analysis_dp_gmm",
                    "0.1.0",
                    now_iso(),
                    None,
                    None,
                    None,
                    "ok",
                    "seed",
                    "{}",
                    "[]",
                    "[]",
                    None,
                ),
            )
        else:
            conn.execute(
                "INSERT INTO plugin_results_v2 (run_id, plugin_id, plugin_version, executed_at, code_hash, settings_hash, dataset_hash, status, summary, metrics_json, findings_json, artifacts_json, error_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    "analysis_dp_gmm",
                    "0.1.0",
                    now_iso(),
                    None,
                    None,
                    None,
                    "ok",
                    "seed",
                    "{}",
                    "[]",
                    "[]",
                    None,
                ),
            )

    if version >= 4:
        conn.execute(
            "INSERT INTO parameter_entities (canonical_text) VALUES (?)",
            ("k=v",),
        )
        conn.execute(
            "INSERT INTO parameter_kv (entity_id, key, value) VALUES (?, ?, ?)",
            (1, "k", "v"),
        )
        conn.execute(
            "INSERT INTO row_parameter_link (dataset_version_id, row_index, entity_id) VALUES (?, ?, ?)",
            ("dv_seed", 0, 1),
        )
        conn.execute(
            "INSERT INTO entities (type, key) VALUES (?, ?)",
            ("dataset_version", "dv_seed"),
        )
        conn.execute(
            "INSERT INTO entities (type, key) VALUES (?, ?)",
            ("parameter", "k=v"),
        )
        conn.execute(
            "INSERT INTO edges (src_entity_id, dst_entity_id, kind, evidence_json, score) VALUES (?, ?, ?, ?, ?)",
            (1, 2, "uses_parameter", json.dumps({"column": "col1"}), None),
        )

    if version >= 5:
        if version >= 16:
            conn.execute(
                "INSERT INTO analysis_jobs (dataset_version_id, tenant_id, plugin_id, plugin_version, code_hash, settings_hash, run_seed, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "dv_seed",
                    DEFAULT_TENANT_ID,
                    "analysis_dp_gmm",
                    "0.1.0",
                    None,
                    None,
                    0,
                    "queued",
                    now_iso(),
                ),
            )
            conn.execute(
                "INSERT INTO deliveries (project_id, tenant_id, dataset_version_id, plugin_id, plugin_version, code_hash, dataset_hash, delivered_at, notes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "project_seed",
                    DEFAULT_TENANT_ID,
                    "dv_seed",
                    "analysis_dp_gmm",
                    "0.1.0",
                    None,
                    "hash",
                    now_iso(),
                    "note",
                ),
            )
        else:
            conn.execute(
                "INSERT INTO analysis_jobs (dataset_version_id, plugin_id, plugin_version, code_hash, settings_hash, run_seed, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "dv_seed",
                    "analysis_dp_gmm",
                    "0.1.0",
                    None,
                    None,
                    0,
                    "queued",
                    now_iso(),
                ),
            )
            conn.execute(
                "INSERT INTO deliveries (project_id, dataset_version_id, plugin_id, plugin_version, code_hash, dataset_hash, delivered_at, notes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "project_seed",
                    "dv_seed",
                    "analysis_dp_gmm",
                    "0.1.0",
                    None,
                    "hash",
                    now_iso(),
                    "note",
                ),
            )

    if version >= 6:
        if version >= 16:
            conn.execute(
                "INSERT INTO raw_formats (fingerprint, tenant_id, name, created_at) VALUES (?, ?, ?, ?)",
                ("raw_fp", DEFAULT_TENANT_ID, "Raw Format", now_iso()),
            )
            conn.execute(
                "INSERT INTO raw_format_notes (format_id, tenant_id, note, created_at) VALUES (?, ?, ?, ?)",
                (1, DEFAULT_TENANT_ID, "note", now_iso()),
            )
        else:
            conn.execute(
                "INSERT INTO raw_formats (fingerprint, name, created_at) VALUES (?, ?, ?)",
                ("raw_fp", "Raw Format", now_iso()),
            )
            conn.execute(
                "INSERT INTO raw_format_notes (format_id, note, created_at) VALUES (?, ?, ?)",
                (1, "note", now_iso()),
            )
        conn.execute(
            "UPDATE dataset_versions SET raw_format_id = ? WHERE dataset_version_id = ?",
            (1, "dv_seed"),
        )
        table_name = "template_seed"
        if version >= 16:
            conn.execute(
                "INSERT INTO templates (name, tenant_id, description, version, created_at, table_name) VALUES (?, ?, ?, ?, ?, ?)",
                ("Template Seed", DEFAULT_TENANT_ID, "desc", "v1", now_iso(), table_name),
            )
            conn.execute(
                "INSERT INTO template_fields (template_id, tenant_id, field_id, safe_name, name, dtype, role, required) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (1, DEFAULT_TENANT_ID, 1, "f1", "value", "float", None, 0),
            )
            conn.execute(
                "INSERT INTO dataset_templates (dataset_version_id, tenant_id, template_id, mapping_json, mapping_hash, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "dv_seed",
                    DEFAULT_TENANT_ID,
                    1,
                    json_dumps({"value": "col1"}),
                    "hash",
                    "ready",
                    now_iso(),
                    now_iso(),
                ),
            )
            conn.execute(
                "INSERT INTO template_conversions (dataset_version_id, tenant_id, template_id, status, started_at, completed_at, error_json, mapping_hash, row_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "dv_seed",
                    DEFAULT_TENANT_ID,
                    1,
                    "completed",
                    now_iso(),
                    now_iso(),
                    None,
                    "hash",
                    1,
                ),
            )
        else:
            conn.execute(
                "INSERT INTO templates (name, description, version, created_at, table_name) VALUES (?, ?, ?, ?, ?)",
                ("Template Seed", "desc", "v1", now_iso(), table_name),
            )
            conn.execute(
                "INSERT INTO template_fields (template_id, field_id, safe_name, name, dtype, role, required) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (1, 1, "f1", "value", "float", None, 0),
            )
            conn.execute(
                "INSERT INTO dataset_templates (dataset_version_id, template_id, mapping_json, mapping_hash, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "dv_seed",
                    1,
                    json_dumps({"value": "col1"}),
                    "hash",
                    "ready",
                    now_iso(),
                    now_iso(),
                ),
            )
            conn.execute(
                "INSERT INTO template_conversions (dataset_version_id, template_id, status, started_at, completed_at, error_json, mapping_hash, row_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("dv_seed", 1, "completed", now_iso(), now_iso(), None, "hash", 1),
            )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS template_seed (row_id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_version_id TEXT, row_index INTEGER, row_json TEXT, f1 TEXT)"
        )

    if version >= 7:
        if version >= 16:
            conn.execute(
                "INSERT INTO raw_format_mappings (format_id, tenant_id, template_id, mapping_json, mapping_hash, notes, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    1,
                    DEFAULT_TENANT_ID,
                    1,
                    json_dumps({"value": "col1"}),
                    "hash",
                    "note",
                    now_iso(),
                ),
            )
        else:
            conn.execute(
                "INSERT INTO raw_format_mappings (format_id, template_id, mapping_json, mapping_hash, notes, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (1, 1, json_dumps({"value": "col1"}), "hash", "note", now_iso()),
            )

    if version >= 8:
        if version >= 16:
            conn.execute(
                "INSERT INTO plugin_executions (run_id, tenant_id, plugin_id, plugin_version, started_at, completed_at, duration_ms, status, exit_code, cpu_user, cpu_system, max_rss, warnings_count, stdout, stderr) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    DEFAULT_TENANT_ID,
                    "analysis_dp_gmm",
                    "0.1.0",
                    now_iso(),
                    now_iso(),
                    1,
                    "ok",
                    0,
                    0.1,
                    0.1,
                    123,
                    0,
                    "out",
                    "",
                ),
            )
        else:
            conn.execute(
                "INSERT INTO plugin_executions (run_id, plugin_id, plugin_version, started_at, completed_at, duration_ms, status, exit_code, cpu_user, cpu_system, max_rss, warnings_count, stdout, stderr) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    "analysis_dp_gmm",
                    "0.1.0",
                    now_iso(),
                    now_iso(),
                    1,
                    "ok",
                    0,
                    0.1,
                    0.1,
                    123,
                    0,
                    "out",
                    "",
                ),
            )

    if version >= 10:
        if version >= 16:
            conn.execute(
                "INSERT INTO known_issue_sets (sha256, tenant_id, upload_id, strict, notes, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "seedhash",
                    DEFAULT_TENANT_ID,
                    "upload_seed",
                    1,
                    "seed notes",
                    now_iso(),
                    now_iso(),
                ),
            )
        else:
            conn.execute(
                "INSERT INTO known_issue_sets (sha256, upload_id, strict, notes, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("seedhash", "upload_seed", 1, "seed notes", now_iso(), now_iso()),
            )
        conn.execute(
            "INSERT INTO known_issues (set_id, title, plugin_id, kind, where_json, contains_json, min_count, max_count, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                1,
                "qemail close cycle",
                "analysis_close_cycle_contention",
                "close_cycle_contention",
                json_dumps({"process": "qemail"}),
                None,
                1,
                1,
                now_iso(),
                now_iso(),
            ),
        )

    if version >= 11:
        if version >= 16:
            conn.execute(
                "INSERT INTO dataset_role_candidates (dataset_version_id, tenant_id, column_id, role, score, reasons_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "dv_seed",
                    DEFAULT_TENANT_ID,
                    1,
                    "start_time",
                    3.0,
                    json_dumps(["name_match"]),
                    now_iso(),
                ),
            )
            conn.execute(
                "INSERT INTO project_role_overrides (project_id, tenant_id, role, column_name, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "project_seed",
                    DEFAULT_TENANT_ID,
                    "queue_time",
                    "col1",
                    now_iso(),
                    now_iso(),
                ),
            )
        else:
            conn.execute(
                "INSERT INTO dataset_role_candidates (dataset_version_id, column_id, role, score, reasons_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "dv_seed",
                    1,
                    "start_time",
                    3.0,
                    json_dumps(["name_match"]),
                    now_iso(),
                ),
            )
            conn.execute(
                "INSERT INTO project_role_overrides (project_id, role, column_name, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("project_seed", "queue_time", "col1", now_iso(), now_iso()),
            )

    conn.commit()


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for version in range(1, len(MIGRATIONS) + 1):
        path = FIXTURE_DIR / f"v{version}.sqlite"
        if path.exists():
            path.unlink()
        conn = sqlite3.connect(path)
        apply_migrations(conn, version)
        seed_base(conn, version)
        conn.close()


if __name__ == "__main__":
    main()
