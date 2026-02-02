from __future__ import annotations

import sqlite3
from typing import Callable

from .utils import DEFAULT_TENANT_ID, now_iso

Migration = Callable[[sqlite3.Connection], None]


def _ensure_schema_migrations(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


def migration_1(conn: sqlite3.Connection) -> None:
    _ensure_schema_migrations(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT,
            status TEXT,
            upload_id TEXT,
            input_filename TEXT,
            canonical_path TEXT,
            settings_json TEXT,
            error_json TEXT,
            run_seed INTEGER DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS plugin_results (
            run_id TEXT,
            plugin_id TEXT,
            status TEXT,
            summary TEXT,
            metrics_json TEXT,
            findings_json TEXT,
            artifacts_json TEXT,
            error_json TEXT,
            PRIMARY KEY (run_id, plugin_id)
        )
        """
    )


def migration_2(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            project_id TEXT PRIMARY KEY,
            fingerprint TEXT UNIQUE NOT NULL,
            name TEXT,
            created_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            fingerprint TEXT UNIQUE NOT NULL,
            created_at TEXT,
            FOREIGN KEY (project_id) REFERENCES projects(project_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_versions (
            dataset_version_id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            created_at TEXT,
            table_name TEXT NOT NULL,
            row_count INTEGER DEFAULT 0,
            column_count INTEGER DEFAULT 0,
            data_hash TEXT,
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_columns (
            dataset_version_id TEXT NOT NULL,
            column_id INTEGER NOT NULL,
            safe_name TEXT NOT NULL,
            original_name TEXT NOT NULL,
            dtype TEXT,
            role TEXT,
            PRIMARY KEY (dataset_version_id, column_id),
            FOREIGN KEY (dataset_version_id) REFERENCES dataset_versions(dataset_version_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS uploads (
            upload_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            created_at TEXT
        )
        """
    )

    if not _column_exists(conn, "runs", "project_id"):
        conn.execute("ALTER TABLE runs ADD COLUMN project_id TEXT")
    if not _column_exists(conn, "runs", "dataset_id"):
        conn.execute("ALTER TABLE runs ADD COLUMN dataset_id TEXT")
    if not _column_exists(conn, "runs", "dataset_version_id"):
        conn.execute("ALTER TABLE runs ADD COLUMN dataset_version_id TEXT")
    if not _column_exists(conn, "runs", "input_hash"):
        conn.execute("ALTER TABLE runs ADD COLUMN input_hash TEXT")

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_dataset_columns_version ON dataset_columns(dataset_version_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_datasets_project ON datasets(project_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_dataset_versions_dataset ON dataset_versions(dataset_id)"
    )


def migration_3(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS plugin_results_v2 (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            plugin_id TEXT NOT NULL,
            plugin_version TEXT,
            executed_at TEXT,
            code_hash TEXT,
            settings_hash TEXT,
            dataset_hash TEXT,
            status TEXT,
            summary TEXT,
            metrics_json TEXT,
            findings_json TEXT,
            artifacts_json TEXT,
            error_json TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_plugin_results_v2_run ON plugin_results_v2(run_id, plugin_id, result_id)"
    )

    cur = conn.execute("SELECT COUNT(*) FROM plugin_results_v2")
    if int(cur.fetchone()[0]) == 0:
        try:
            legacy_count = conn.execute("SELECT COUNT(*) FROM plugin_results").fetchone()
            if legacy_count and int(legacy_count[0]) > 0:
                conn.execute(
                    """
                    INSERT INTO plugin_results_v2
                    (run_id, plugin_id, status, summary, metrics_json, findings_json, artifacts_json, error_json, executed_at)
                    SELECT run_id, plugin_id, status, summary, metrics_json, findings_json, artifacts_json, error_json, ?
                    FROM plugin_results
                    """,
                    (now_iso(),),
                )
        except sqlite3.OperationalError:
            pass


def migration_4(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS parameter_entities (
            entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_text TEXT UNIQUE NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS parameter_kv (
            entity_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value TEXT,
            PRIMARY KEY (entity_id, key, value),
            FOREIGN KEY (entity_id) REFERENCES parameter_entities(entity_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS row_parameter_link (
            dataset_version_id TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            entity_id INTEGER NOT NULL,
            PRIMARY KEY (dataset_version_id, row_index, entity_id),
            FOREIGN KEY (entity_id) REFERENCES parameter_entities(entity_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_parameter_kv ON parameter_kv(key, value)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_row_parameter_link ON row_parameter_link(dataset_version_id, entity_id)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS entities (
            entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            key TEXT NOT NULL,
            UNIQUE (type, key)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS edges (
            edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
            src_entity_id INTEGER NOT NULL,
            dst_entity_id INTEGER NOT NULL,
            kind TEXT NOT NULL,
            evidence_json TEXT,
            score REAL,
            FOREIGN KEY (src_entity_id) REFERENCES entities(entity_id),
            FOREIGN KEY (dst_entity_id) REFERENCES entities(entity_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_entities_type_key ON entities(type, key)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_entity_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_entity_id)"
    )


def migration_5(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_jobs (
            job_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_version_id TEXT NOT NULL,
            plugin_id TEXT NOT NULL,
            plugin_version TEXT,
            code_hash TEXT,
            settings_hash TEXT,
            run_seed INTEGER DEFAULT 0,
            status TEXT NOT NULL,
            created_at TEXT,
            started_at TEXT,
            completed_at TEXT,
            error_json TEXT,
            UNIQUE (dataset_version_id, plugin_id, plugin_version, code_hash, settings_hash)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status ON analysis_jobs(status)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS deliveries (
            delivery_id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            dataset_version_id TEXT NOT NULL,
            plugin_id TEXT NOT NULL,
            plugin_version TEXT,
            code_hash TEXT,
            dataset_hash TEXT,
            delivered_at TEXT,
            notes TEXT,
            UNIQUE (dataset_version_id, plugin_id, plugin_version, code_hash, dataset_hash)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_deliveries_dataset ON deliveries(dataset_version_id, plugin_id)"
    )


def migration_6(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_formats (
            format_id INTEGER PRIMARY KEY AUTOINCREMENT,
            fingerprint TEXT UNIQUE NOT NULL,
            name TEXT,
            created_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_format_notes (
            note_id INTEGER PRIMARY KEY AUTOINCREMENT,
            format_id INTEGER NOT NULL,
            note TEXT NOT NULL,
            created_at TEXT,
            FOREIGN KEY (format_id) REFERENCES raw_formats(format_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS templates (
            template_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            version TEXT,
            created_at TEXT,
            table_name TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS template_fields (
            template_id INTEGER NOT NULL,
            field_id INTEGER NOT NULL,
            safe_name TEXT NOT NULL,
            name TEXT NOT NULL,
            dtype TEXT,
            role TEXT,
            required INTEGER DEFAULT 0,
            PRIMARY KEY (template_id, field_id),
            UNIQUE (template_id, name),
            FOREIGN KEY (template_id) REFERENCES templates(template_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_templates (
            dataset_version_id TEXT NOT NULL,
            template_id INTEGER NOT NULL,
            mapping_json TEXT NOT NULL,
            mapping_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT,
            updated_at TEXT,
            PRIMARY KEY (dataset_version_id, template_id),
            FOREIGN KEY (template_id) REFERENCES templates(template_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS template_conversions (
            conversion_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_version_id TEXT NOT NULL,
            template_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            error_json TEXT,
            mapping_hash TEXT NOT NULL,
            row_count INTEGER DEFAULT 0,
            FOREIGN KEY (template_id) REFERENCES templates(template_id)
        )
        """
    )
    if not _column_exists(conn, "dataset_versions", "raw_format_id"):
        conn.execute("ALTER TABLE dataset_versions ADD COLUMN raw_format_id INTEGER")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_dataset_templates_status ON dataset_templates(status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_template_fields_template ON template_fields(template_id)"
    )


def migration_7(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_format_mappings (
            mapping_id INTEGER PRIMARY KEY AUTOINCREMENT,
            format_id INTEGER NOT NULL,
            template_id INTEGER NOT NULL,
            mapping_json TEXT NOT NULL,
            mapping_hash TEXT NOT NULL,
            notes TEXT,
            created_at TEXT,
            UNIQUE (format_id, template_id, mapping_hash),
            FOREIGN KEY (format_id) REFERENCES raw_formats(format_id),
            FOREIGN KEY (template_id) REFERENCES templates(template_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_raw_format_mappings_format ON raw_format_mappings(format_id, template_id)"
    )


def migration_8(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS plugin_executions (
            execution_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            plugin_id TEXT NOT NULL,
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
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_plugin_executions_run ON plugin_executions(run_id, plugin_id)"
    )


def migration_9(conn: sqlite3.Connection) -> None:
    if not _column_exists(conn, "runs", "run_seed"):
        conn.execute("ALTER TABLE runs ADD COLUMN run_seed INTEGER DEFAULT 0")


def migration_10(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS known_issue_sets (
            set_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sha256 TEXT NOT NULL UNIQUE,
            upload_id TEXT,
            strict INTEGER DEFAULT 1,
            notes TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS known_issues (
            issue_id INTEGER PRIMARY KEY AUTOINCREMENT,
            set_id INTEGER NOT NULL,
            title TEXT,
            plugin_id TEXT,
            kind TEXT NOT NULL,
            where_json TEXT,
            contains_json TEXT,
            min_count INTEGER,
            max_count INTEGER,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY (set_id) REFERENCES known_issue_sets(set_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_known_issues_set ON known_issues(set_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_known_issues_kind ON known_issues(kind)"
    )


def migration_11(conn: sqlite3.Connection) -> None:
    if not _column_exists(conn, "projects", "erp_type"):
        conn.execute("ALTER TABLE projects ADD COLUMN erp_type TEXT")
    conn.execute(
        "UPDATE projects SET erp_type = 'unknown' WHERE erp_type IS NULL OR erp_type = ''"
    )

    if not _column_exists(conn, "known_issue_sets", "scope_type"):
        conn.execute("ALTER TABLE known_issue_sets ADD COLUMN scope_type TEXT")
    if not _column_exists(conn, "known_issue_sets", "scope_value"):
        conn.execute("ALTER TABLE known_issue_sets ADD COLUMN scope_value TEXT")
    conn.execute(
        "UPDATE known_issue_sets SET scope_type = 'sha256', scope_value = sha256 "
        "WHERE scope_type IS NULL OR scope_value IS NULL"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_role_candidates (
            dataset_version_id TEXT NOT NULL,
            column_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            score REAL NOT NULL,
            reasons_json TEXT,
            created_at TEXT,
            PRIMARY KEY (dataset_version_id, column_id, role)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_dataset_role_candidates "
        "ON dataset_role_candidates(dataset_version_id, role, score)"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS project_role_overrides (
            project_id TEXT NOT NULL,
            role TEXT NOT NULL,
            column_name TEXT NOT NULL,
            created_at TEXT,
            updated_at TEXT,
            PRIMARY KEY (project_id, role),
            FOREIGN KEY (project_id) REFERENCES projects(project_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_project_role_overrides_project "
        "ON project_role_overrides(project_id)"
    )


def migration_12(conn: sqlite3.Connection) -> None:
    if not _column_exists(conn, "known_issues", "description"):
        conn.execute("ALTER TABLE known_issues ADD COLUMN description TEXT")


def migration_13(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS project_plugin_settings (
            project_id TEXT NOT NULL,
            plugin_id TEXT NOT NULL,
            settings_json TEXT,
            created_at TEXT,
            updated_at TEXT,
            PRIMARY KEY (project_id, plugin_id),
            FOREIGN KEY (project_id) REFERENCES projects(project_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_project_plugin_settings_project "
        "ON project_plugin_settings(project_id)"
    )


def migration_14(conn: sqlite3.Connection) -> None:
    if not _column_exists(conn, "plugin_results_v2", "budget_json"):
        conn.execute("ALTER TABLE plugin_results_v2 ADD COLUMN budget_json TEXT")


def migration_15(conn: sqlite3.Connection) -> None:
    if not _column_exists(conn, "dataset_columns", "pii_tags_json"):
        conn.execute("ALTER TABLE dataset_columns ADD COLUMN pii_tags_json TEXT")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pii_salts (
            tenant_id TEXT PRIMARY KEY,
            salt TEXT NOT NULL,
            created_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pii_entities (
            entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id TEXT NOT NULL,
            pii_type TEXT NOT NULL,
            raw_value TEXT NOT NULL,
            value_hash TEXT NOT NULL,
            created_at TEXT,
            UNIQUE (tenant_id, pii_type, raw_value)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pii_entities_hash ON pii_entities(tenant_id, value_hash)"
    )


def migration_16(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tenants (
            tenant_id TEXT PRIMARY KEY,
            name TEXT,
            created_at TEXT,
            is_default INTEGER DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO tenants (tenant_id, name, created_at, is_default)
        VALUES (?, ?, ?, 1)
        """,
        (DEFAULT_TENANT_ID, "Default", now_iso()),
    )

    tenant_tables = [
        "projects",
        "datasets",
        "dataset_versions",
        "dataset_columns",
        "uploads",
        "runs",
        "plugin_results_v2",
        "plugin_executions",
        "known_issue_sets",
        "dataset_role_candidates",
        "project_role_overrides",
        "project_plugin_settings",
        "analysis_jobs",
        "deliveries",
        "raw_formats",
        "raw_format_notes",
        "raw_format_mappings",
        "templates",
        "template_fields",
        "dataset_templates",
        "template_conversions",
    ]
    for table in tenant_tables:
        if not _column_exists(conn, table, "tenant_id"):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN tenant_id TEXT")
        conn.execute(
            f"UPDATE {table} SET tenant_id = ? WHERE tenant_id IS NULL OR tenant_id = ''",
            (DEFAULT_TENANT_ID,),
        )

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_projects_tenant ON projects(tenant_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_datasets_tenant ON datasets(tenant_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_dataset_versions_tenant ON dataset_versions(tenant_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_runs_tenant ON runs(tenant_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_uploads_tenant ON uploads(tenant_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_plugin_results_tenant ON plugin_results_v2(tenant_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_known_issue_sets_tenant ON known_issue_sets(tenant_id)"
    )


def migration_17(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT,
            password_hash TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TEXT,
            disabled_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tenant_memberships (
            membership_id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            created_at TEXT,
            UNIQUE (tenant_id, user_id),
            FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id),
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            tenant_id TEXT NOT NULL,
            token_hash TEXT UNIQUE NOT NULL,
            created_at TEXT,
            last_seen_at TEXT,
            expires_at TEXT,
            revoked_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS api_keys (
            key_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            tenant_id TEXT NOT NULL,
            name TEXT,
            key_hash TEXT UNIQUE NOT NULL,
            created_at TEXT,
            last_used_at TEXT,
            revoked_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memberships_tenant ON tenant_memberships(tenant_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memberships_user ON tenant_memberships(user_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(token_hash)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_user ON user_sessions(user_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash)"
    )


def migration_18(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS vector_collections (
            collection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id TEXT NOT NULL,
            name TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            table_name TEXT NOT NULL,
            created_at TEXT,
            UNIQUE (tenant_id, name, dimensions)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_vector_collections_tenant ON vector_collections(tenant_id)"
    )


MIGRATIONS: list[Migration] = [
    migration_1,
    migration_2,
    migration_3,
    migration_4,
    migration_5,
    migration_6,
    migration_7,
    migration_8,
    migration_9,
    migration_10,
    migration_11,
    migration_12,
    migration_13,
    migration_14,
    migration_15,
    migration_16,
    migration_17,
    migration_18,
]


def run_migrations(conn: sqlite3.Connection) -> None:
    _ensure_schema_migrations(conn)
    cur = conn.execute("PRAGMA user_version")
    current = int(cur.fetchone()[0])
    for version, migration in enumerate(MIGRATIONS, start=1):
        if version <= current:
            continue
        migration(conn)
        conn.execute(
            "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            (version, now_iso()),
        )
        conn.execute(f"PRAGMA user_version = {version}")
