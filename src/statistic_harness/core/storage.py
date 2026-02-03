from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .migrations import run_migrations
from .types import PluginResult
from .utils import (
    DEFAULT_TENANT_ID,
    ensure_dir,
    json_dumps,
    now_iso,
    quote_identifier,
    scope_key,
)


class Storage:
    def __init__(self, db_path: Path, tenant_id: str | None = None) -> None:
        ensure_dir(db_path.parent)
        self.db_path = db_path
        self.tenant_id = tenant_id or DEFAULT_TENANT_ID
        with self.connection() as conn:
            run_migrations(conn)

    def _tenant_id(self) -> str:
        return self.tenant_id or DEFAULT_TENANT_ID

    def _scoped_key(self, scope_type: str, scope_value: str) -> str:
        tenant_id = self._tenant_id()
        if tenant_id == DEFAULT_TENANT_ID:
            return scope_key(scope_type, scope_value)
        return scope_key(f"{tenant_id}:{scope_type}", scope_value)

    def _scoped_value(self, value: str) -> str:
        tenant_id = self._tenant_id()
        if tenant_id == DEFAULT_TENANT_ID:
            return value
        prefix = f"{tenant_id}__"
        if value.startswith(prefix):
            return value
        return f"{prefix}{value}"

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA temp_store = MEMORY")
        return conn

    @contextmanager
    def connection(self) -> Iterable[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create_run(
        self,
        run_id: str,
        created_at: str,
        status: str,
        upload_id: str,
        input_filename: str,
        canonical_path: str,
        settings: dict[str, Any],
        error: dict[str, Any] | None,
        run_seed: int = 0,
        project_id: str | None = None,
        dataset_id: str | None = None,
        dataset_version_id: str | None = None,
        input_hash: str | None = None,
    ) -> None:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO runs
                (run_id, tenant_id, created_at, status, upload_id, input_filename, canonical_path, settings_json, error_json,
                 run_seed, project_id, dataset_id, dataset_version_id, input_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    tenant_id,
                    created_at,
                    status,
                    upload_id,
                    input_filename,
                    canonical_path,
                    json_dumps(settings),
                    json_dumps(error) if error else None,
                    int(run_seed),
                    project_id,
                    dataset_id,
                    dataset_version_id,
                    input_hash,
                ),
            )

    def update_run_status(
        self, run_id: str, status: str, error: dict[str, Any] | None = None
    ) -> None:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, error_json = ? WHERE run_id = ? AND tenant_id = ?",
                (status, json_dumps(error) if error else None, run_id, tenant_id),
            )

    def save_plugin_result(
        self,
        run_id: str,
        plugin_id: str,
        plugin_version: str | None,
        executed_at: str,
        code_hash: str | None,
        settings_hash: str | None,
        dataset_hash: str | None,
        result: PluginResult,
    ) -> None:
        tenant_id = self._tenant_id()
        artifacts = [asdict(a) for a in result.artifacts]
        error_payload = asdict(result.error) if result.error else None
        budget_payload = result.budget if isinstance(result.budget, dict) else {}
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO plugin_results_v2
                (run_id, tenant_id, plugin_id, plugin_version, executed_at, code_hash, settings_hash, dataset_hash,
                 status, summary, metrics_json, findings_json, artifacts_json, error_json, budget_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    tenant_id,
                    plugin_id,
                    plugin_version,
                    executed_at,
                    code_hash,
                    settings_hash,
                    dataset_hash,
                    result.status,
                    result.summary,
                    json_dumps(result.metrics),
                    json_dumps(result.findings),
                    json_dumps(artifacts),
                    json_dumps(error_payload) if error_payload else None,
                    json_dumps(budget_payload),
                ),
            )

    def insert_plugin_execution(
        self,
        run_id: str,
        plugin_id: str,
        plugin_version: str | None,
        started_at: str | None,
        completed_at: str | None,
        duration_ms: int | None,
        status: str,
        exit_code: int | None,
        cpu_user: float | None,
        cpu_system: float | None,
        max_rss: int | None,
        warnings_count: int | None,
        stdout: str | None,
        stderr: str | None,
    ) -> None:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO plugin_executions
                (run_id, tenant_id, plugin_id, plugin_version, started_at, completed_at, duration_ms,
                 status, exit_code, cpu_user, cpu_system, max_rss, warnings_count, stdout, stderr)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    tenant_id,
                    plugin_id,
                    plugin_version,
                    started_at,
                    completed_at,
                    duration_ms,
                    status,
                    exit_code,
                    cpu_user,
                    cpu_system,
                    max_rss,
                    warnings_count,
                    stdout,
                    stderr,
                ),
            )

    def start_plugin_execution(
        self,
        run_id: str,
        plugin_id: str,
        plugin_version: str | None,
        started_at: str,
        status: str = "running",
    ) -> int:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO plugin_executions
                (run_id, tenant_id, plugin_id, plugin_version, started_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, tenant_id, plugin_id, plugin_version, started_at, status),
            )
            return int(cur.lastrowid)

    def update_plugin_execution(
        self,
        execution_id: int,
        completed_at: str | None,
        duration_ms: int | None,
        status: str,
        exit_code: int | None,
        cpu_user: float | None,
        cpu_system: float | None,
        max_rss: int | None,
        warnings_count: int | None,
        stdout: str | None,
        stderr: str | None,
    ) -> None:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE plugin_executions
                SET completed_at = ?, duration_ms = ?, status = ?, exit_code = ?,
                    cpu_user = ?, cpu_system = ?, max_rss = ?, warnings_count = ?,
                    stdout = ?, stderr = ?
                WHERE execution_id = ? AND tenant_id = ?
                """,
                (
                    completed_at,
                    duration_ms,
                    status,
                    exit_code,
                    cpu_user,
                    cpu_system,
                    max_rss,
                    warnings_count,
                    stdout,
                    stderr,
                    execution_id,
                    tenant_id,
                ),
            )

    def fetch_plugin_executions(self, run_id: str) -> list[dict[str, Any]]:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT *
                FROM plugin_executions
                WHERE run_id = ? AND tenant_id = ?
                ORDER BY execution_id ASC
                """,
                (run_id, tenant_id),
            )
            return [dict(row) for row in cur.fetchall()]

    def fetch_plugin_results(self, run_id: str) -> list[dict[str, Any]]:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT pr.*
                FROM plugin_results_v2 pr
                JOIN (
                    SELECT plugin_id, MAX(result_id) AS max_id
                    FROM plugin_results_v2
                    WHERE run_id = ? AND tenant_id = ?
                    GROUP BY plugin_id
                ) latest
                ON pr.plugin_id = latest.plugin_id AND pr.result_id = latest.max_id
                WHERE pr.run_id = ? AND pr.tenant_id = ?
                """,
                (run_id, tenant_id, run_id, tenant_id),
            )
            return [dict(row) for row in cur.fetchall()]

    def fetch_run(self, run_id: str) -> dict[str, Any] | None:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                "SELECT * FROM runs WHERE run_id = ? AND tenant_id = ?",
                (run_id, tenant_id),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def fetch_upload(self, upload_id: str) -> dict[str, Any] | None:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                "SELECT * FROM uploads WHERE upload_id = ? AND tenant_id = ?",
                (upload_id, tenant_id),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def fetch_upload_by_sha256(self, sha256: str) -> dict[str, Any] | None:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                "SELECT * FROM uploads WHERE sha256 = ? AND tenant_id = ? ORDER BY created_at ASC LIMIT 1",
                (sha256, tenant_id),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def list_uploads(self, limit: int = 50) -> list[dict[str, Any]]:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT upload_id, filename, size_bytes, sha256, created_at
                FROM uploads
                WHERE tenant_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (tenant_id, int(limit)),
            )
            return [dict(row) for row in cur.fetchall()]

    def fetch_known_issues(
        self, scope_value: str, scope_type: str = "sha256"
    ) -> dict[str, Any] | None:
        tenant_id = self._tenant_id()
        key = self._scoped_key(scope_type, scope_value)
        with self.connection() as conn:
            cur = conn.execute(
                "SELECT * FROM known_issue_sets WHERE sha256 = ? AND tenant_id = ?",
                (key, tenant_id),
            )
            set_row = cur.fetchone()
            if not set_row:
                return None
            set_row = dict(set_row)
            cur = conn.execute(
                """
                SELECT * FROM known_issues
                WHERE set_id = ?
                ORDER BY issue_id
                """,
                (set_row["set_id"],),
            )
            issues = []
            for row in cur.fetchall():
                row = dict(row)
                entry: dict[str, Any] = {"kind": row["kind"]}
                if row.get("plugin_id"):
                    entry["plugin_id"] = row["plugin_id"]
                if row.get("title"):
                    entry["title"] = row["title"]
                if row.get("description"):
                    entry["description"] = row["description"]
                if row.get("source_text"):
                    entry["source_text"] = row["source_text"]
                if row.get("where_json"):
                    try:
                        entry["where"] = json.loads(row["where_json"])
                    except json.JSONDecodeError:
                        pass
                if row.get("contains_json"):
                    try:
                        entry["contains"] = json.loads(row["contains_json"])
                    except json.JSONDecodeError:
                        pass
                if row.get("min_count") is not None:
                    entry["min_count"] = int(row["min_count"])
                if row.get("max_count") is not None:
                    entry["max_count"] = int(row["max_count"])
                issues.append(entry)
            return {
                "set_id": int(set_row["set_id"]),
                "sha256": set_row.get("sha256") or "",
                "scope_type": set_row.get("scope_type") or "sha256",
                "scope_value": set_row.get("scope_value") or set_row.get("sha256") or "",
                "upload_id": set_row.get("upload_id"),
                "strict": bool(set_row.get("strict", 0)),
                "notes": set_row.get("notes") or "",
                "natural_language": json.loads(set_row["nl_json"])
                if set_row.get("nl_json")
                else [],
                "expected_findings": issues,
            }

    def upsert_known_issue_set(
        self,
        scope_value: str,
        scope_type: str,
        upload_id: str | None,
        strict: bool,
        notes: str,
        natural_language: list[dict[str, Any]] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> int:
        tenant_id = self._tenant_id()
        key = self._scoped_key(scope_type, scope_value)
        if conn is None:
            with self.connection() as temp:
                return self.upsert_known_issue_set(
                    scope_value,
                    scope_type,
                    upload_id,
                    strict,
                    notes,
                    natural_language,
                    temp,
                )
        created_at = now_iso()
        updated_at = created_at
        nl_json = json_dumps(natural_language or []) if natural_language else None
        conn.execute(
            """
            INSERT INTO known_issue_sets
            (sha256, tenant_id, upload_id, strict, notes, created_at, updated_at, scope_type, scope_value, nl_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sha256) DO UPDATE SET
              tenant_id = excluded.tenant_id,
              upload_id = excluded.upload_id,
              strict = excluded.strict,
              notes = excluded.notes,
              updated_at = excluded.updated_at,
              scope_type = excluded.scope_type,
              scope_value = excluded.scope_value,
              nl_json = excluded.nl_json
            """,
            (
                key,
                tenant_id,
                upload_id,
                1 if strict else 0,
                notes,
                created_at,
                updated_at,
                scope_type,
                scope_value,
                nl_json,
            ),
        )
        cur = conn.execute(
            "SELECT set_id FROM known_issue_sets WHERE sha256 = ? AND tenant_id = ?",
            (key, tenant_id),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError("Known issue set not found after upsert")
        return int(row["set_id"])

    def replace_known_issues(
        self,
        set_id: int,
        issues: list[dict[str, Any]],
        conn: sqlite3.Connection | None = None,
    ) -> None:
        if conn is None:
            with self.connection() as temp:
                self.replace_known_issues(set_id, issues, temp)
                return
        conn.execute("DELETE FROM known_issues WHERE set_id = ?", (set_id,))
        if not issues:
            return
        created_at = now_iso()
        rows = []
        for issue in issues:
            rows.append(
                (
                    set_id,
                    issue.get("title"),
                    issue.get("description"),
                    issue.get("plugin_id"),
                    issue.get("kind"),
                    json_dumps(issue.get("where")) if issue.get("where") else None,
                    json_dumps(issue.get("contains")) if issue.get("contains") else None,
                    issue.get("min_count"),
                    issue.get("max_count"),
                    issue.get("source_text"),
                    created_at,
                    created_at,
                )
            )
        conn.executemany(
            """
            INSERT INTO known_issues
            (set_id, title, description, plugin_id, kind, where_json, contains_json, min_count, max_count, source_text, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def get_or_create_parameter_entity(
        self, canonical_text: str, conn: sqlite3.Connection | None = None
    ) -> int:
        if conn is None:
            with self.connection() as temp:
                return self.get_or_create_parameter_entity(canonical_text, temp)
        conn.execute(
            "INSERT OR IGNORE INTO parameter_entities (canonical_text) VALUES (?)",
            (canonical_text,),
        )
        cur = conn.execute(
            "SELECT entity_id FROM parameter_entities WHERE canonical_text = ?",
            (canonical_text,),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError("Parameter entity not found")
        return int(row[0])

    def insert_parameter_kv(
        self,
        entity_id: int,
        kv_pairs: list[tuple[str, str]],
        conn: sqlite3.Connection | None = None,
    ) -> None:
        if not kv_pairs:
            return
        if conn is None:
            with self.connection() as temp:
                self.insert_parameter_kv(entity_id, kv_pairs, temp)
                return
        conn.executemany(
            """
            INSERT OR IGNORE INTO parameter_kv (entity_id, key, value)
            VALUES (?, ?, ?)
            """,
            [(entity_id, key, value) for key, value in kv_pairs],
        )

    def insert_row_parameter_links(
        self,
        dataset_version_id: str,
        links: list[tuple[int, int]],
        conn: sqlite3.Connection | None = None,
    ) -> None:
        if not links:
            return
        if conn is None:
            with self.connection() as temp:
                self.insert_row_parameter_links(dataset_version_id, links, temp)
                return
        conn.executemany(
            """
            INSERT OR IGNORE INTO row_parameter_link
            (dataset_version_id, row_index, entity_id)
            VALUES (?, ?, ?)
            """,
            [(dataset_version_id, row_index, entity_id) for row_index, entity_id in links],
        )

    def ensure_entity(
        self, entity_type: str, key: str, conn: sqlite3.Connection | None = None
    ) -> int:
        if conn is None:
            with self.connection() as temp:
                return self.ensure_entity(entity_type, key, temp)
        conn.execute(
            "INSERT OR IGNORE INTO entities (type, key) VALUES (?, ?)",
            (entity_type, key),
        )
        cur = conn.execute(
            "SELECT entity_id FROM entities WHERE type = ? AND key = ?",
            (entity_type, key),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError("Entity not found")
        return int(row[0])

    def add_edges(
        self,
        edges: list[tuple[int, int, str, dict[str, Any] | None, float | None]],
        conn: sqlite3.Connection | None = None,
    ) -> None:
        if not edges:
            return
        if conn is None:
            with self.connection() as temp:
                self.add_edges(edges, temp)
                return
        conn.executemany(
            """
            INSERT INTO edges
            (src_entity_id, dst_entity_id, kind, evidence_json, score)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    src,
                    dst,
                    kind,
                    json_dumps(evidence) if evidence else None,
                    score,
                )
                for src, dst, kind, evidence, score in edges
            ],
        )

    def create_upload(
        self,
        upload_id: str,
        filename: str,
        size_bytes: int,
        sha256: str,
        created_at: str,
    ) -> None:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO uploads
                (upload_id, tenant_id, filename, size_bytes, sha256, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (upload_id, tenant_id, filename, size_bytes, sha256, created_at),
            )

    def ensure_project(self, project_id: str, fingerprint: str, created_at: str) -> None:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO projects
                (project_id, tenant_id, fingerprint, created_at, erp_type)
                VALUES (?, ?, ?, ?, ?)
                """,
                (project_id, tenant_id, fingerprint, created_at, "unknown"),
            )

    def ensure_dataset(
        self, dataset_id: str, project_id: str, fingerprint: str, created_at: str
    ) -> None:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO datasets
                (dataset_id, tenant_id, project_id, fingerprint, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (dataset_id, tenant_id, project_id, fingerprint, created_at),
            )

    def get_dataset_version(
        self, dataset_version_id: str, conn: sqlite3.Connection | None = None
    ) -> dict[str, Any] | None:
        tenant_id = self._tenant_id()
        if conn is None:
            with self.connection() as temp:
                return self.get_dataset_version(dataset_version_id, temp)
        cur = conn.execute(
            "SELECT * FROM dataset_versions WHERE dataset_version_id = ? AND tenant_id = ?",
            (dataset_version_id, tenant_id),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_dataset_version_context(
        self, dataset_version_id: str, conn: sqlite3.Connection | None = None
    ) -> dict[str, Any] | None:
        tenant_id = self._tenant_id()
        if conn is None:
            with self.connection() as temp:
                return self.get_dataset_version_context(dataset_version_id, temp)
        cur = conn.execute(
            """
            SELECT dv.dataset_version_id,
                   dv.dataset_id,
                   dv.table_name,
                   dv.data_hash,
                   d.project_id
            FROM dataset_versions dv
            JOIN datasets d ON d.dataset_id = dv.dataset_id
            WHERE dv.dataset_version_id = ? AND dv.tenant_id = ? AND d.tenant_id = ?
            """,
            (dataset_version_id, tenant_id, tenant_id),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def ensure_dataset_version(
        self,
        dataset_version_id: str,
        dataset_id: str,
        created_at: str,
        table_name: str,
        data_hash: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        tenant_id = self._tenant_id()
        if conn is None:
            with self.connection() as temp:
                self.ensure_dataset_version(
                    dataset_version_id,
                    dataset_id,
                    created_at,
                    table_name,
                    data_hash,
                    temp,
                )
                return
        conn.execute(
            """
            INSERT OR IGNORE INTO dataset_versions
            (dataset_version_id, tenant_id, dataset_id, created_at, table_name, data_hash)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (dataset_version_id, tenant_id, dataset_id, created_at, table_name, data_hash),
        )

    def update_dataset_version_stats(
        self,
        dataset_version_id: str,
        row_count: int,
        column_count: int,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        tenant_id = self._tenant_id()
        if conn is None:
            with self.connection() as temp:
                self.update_dataset_version_stats(
                    dataset_version_id, row_count, column_count, temp
                )
                return
        conn.execute(
            """
            UPDATE dataset_versions
            SET row_count = ?, column_count = ?
            WHERE dataset_version_id = ? AND tenant_id = ?
            """,
            (row_count, column_count, dataset_version_id, tenant_id),
        )

    def set_dataset_raw_format(
        self,
        dataset_version_id: str,
        format_id: int,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        tenant_id = self._tenant_id()
        if conn is None:
            with self.connection() as temp:
                self.set_dataset_raw_format(dataset_version_id, format_id, temp)
                return
        conn.execute(
            """
            UPDATE dataset_versions
            SET raw_format_id = ?
            WHERE dataset_version_id = ? AND tenant_id = ?
            """,
            (format_id, dataset_version_id, tenant_id),
        )

    def replace_dataset_columns(
        self,
        dataset_version_id: str,
        columns: list[dict[str, Any]],
        conn: sqlite3.Connection | None = None,
    ) -> None:
        tenant_id = self._tenant_id()
        if conn is None:
            with self.connection() as temp:
                self.replace_dataset_columns(dataset_version_id, columns, temp)
                return
        conn.execute(
            "DELETE FROM dataset_columns WHERE dataset_version_id = ? AND tenant_id = ?",
            (dataset_version_id, tenant_id),
        )
        conn.executemany(
            """
            INSERT INTO dataset_columns
            (dataset_version_id, tenant_id, column_id, safe_name, original_name, dtype, role, pii_tags_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    dataset_version_id,
                    tenant_id,
                    col["column_id"],
                    col["safe_name"],
                    col["original_name"],
                    col.get("dtype"),
                    col.get("role"),
                    json_dumps(col.get("pii_tags")) if col.get("pii_tags") else None,
                )
                for col in columns
            ],
        )

    def update_dataset_column_roles(
        self,
        dataset_version_id: str,
        role_by_name: dict[str, str],
        conn: sqlite3.Connection | None = None,
    ) -> None:
        tenant_id = self._tenant_id()
        if not role_by_name:
            return
        if conn is None:
            with self.connection() as temp:
                self.update_dataset_column_roles(dataset_version_id, role_by_name, temp)
                return
        conn.executemany(
            """
            UPDATE dataset_columns
            SET role = ?
            WHERE dataset_version_id = ? AND original_name = ? AND tenant_id = ?
            """,
            [
                (role, dataset_version_id, name, tenant_id)
                for name, role in role_by_name.items()
            ],
        )

    def fetch_dataset_columns(
        self, dataset_version_id: str, conn: sqlite3.Connection | None = None
    ) -> list[dict[str, Any]]:
        tenant_id = self._tenant_id()
        if conn is None:
            with self.connection() as temp:
                return self.fetch_dataset_columns(dataset_version_id, temp)
        cur = conn.execute(
            """
            SELECT column_id, safe_name, original_name, dtype, role, pii_tags_json
            FROM dataset_columns
            WHERE dataset_version_id = ? AND tenant_id = ?
            ORDER BY column_id
            """,
            (dataset_version_id, tenant_id),
        )
        rows = []
        for row in cur.fetchall():
            entry = dict(row)
            tags = entry.get("pii_tags_json")
            if tags:
                try:
                    entry["pii_tags"] = json.loads(tags)
                except json.JSONDecodeError:
                    entry["pii_tags"] = []
            rows.append(entry)
        return rows

    def update_dataset_column_pii_tags(
        self,
        dataset_version_id: str,
        tags_by_name: dict[str, list[str]],
        conn: sqlite3.Connection | None = None,
    ) -> None:
        tenant_id = self._tenant_id()
        if not tags_by_name:
            return
        if conn is None:
            with self.connection() as temp:
                self.update_dataset_column_pii_tags(
                    dataset_version_id, tags_by_name, temp
                )
                return
        conn.executemany(
            """
            UPDATE dataset_columns
            SET pii_tags_json = ?
            WHERE dataset_version_id = ? AND original_name = ? AND tenant_id = ?
            """,
            [
                (json_dumps(tags), dataset_version_id, name, tenant_id)
                for name, tags in tags_by_name.items()
            ],
        )

    def ensure_pii_salt(self, tenant_id: str) -> str:
        with self.connection() as conn:
            cur = conn.execute(
                "SELECT salt FROM pii_salts WHERE tenant_id = ?",
                (tenant_id,),
            )
            row = cur.fetchone()
            if row and row["salt"]:
                return str(row["salt"])
            salt = hashlib.sha256(
                f"{tenant_id}:{now_iso()}".encode("utf-8")
            ).hexdigest()
            conn.execute(
                "INSERT OR REPLACE INTO pii_salts (tenant_id, salt, created_at) VALUES (?, ?, ?)",
                (tenant_id, salt, now_iso()),
            )
            return salt

    def upsert_pii_entities(
        self,
        tenant_id: str,
        pii_type: str,
        raw_values: list[str],
        conn: sqlite3.Connection | None = None,
    ) -> None:
        if not raw_values:
            return
        if conn is None:
            with self.connection() as temp:
                self.upsert_pii_entities(tenant_id, pii_type, raw_values, temp)
                return
        salt = self.ensure_pii_salt(tenant_id)
        rows = []
        for value in raw_values:
            value = str(value)
            value_hash = hashlib.sha256(
                f"{salt}:{value}".encode("utf-8")
            ).hexdigest()
            rows.append((tenant_id, pii_type, value, value_hash, now_iso()))
        conn.executemany(
            """
            INSERT OR IGNORE INTO pii_entities
            (tenant_id, pii_type, raw_value, value_hash, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )

    def fetch_pii_entities(self, tenant_id: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            cur = conn.execute(
                "SELECT pii_type, raw_value, value_hash FROM pii_entities WHERE tenant_id = ?",
                (tenant_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def fetch_row_trace(
        self,
        dataset_version_id: str,
        row_index: int,
        source_dataset_version_id: str | None = None,
        max_rows: int = 50,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            version = self.get_dataset_version(dataset_version_id, conn)
            if not version:
                raise ValueError("Dataset version not found")
            table_name = version["table_name"]
            dataset_template = self.fetch_dataset_template(dataset_version_id)
            template_fields: list[dict[str, Any]] | None = None
            name_lookup: dict[str, str] = {}
            if dataset_template:
                template_id = int(dataset_template["template_id"])
                template_fields = self.fetch_template_fields(template_id)
                name_lookup = {
                    field["safe_name"]: field["name"] for field in template_fields
                }
            else:
                columns = self.fetch_dataset_columns(dataset_version_id, conn)
                name_lookup = {
                    col["safe_name"]: col["original_name"] for col in columns
                }

            col_info = conn.execute(
                f"PRAGMA table_info({quote_identifier(table_name)})"
            ).fetchall()
            table_cols = [row["name"] for row in col_info]
            has_dataset_id = "dataset_version_id" in table_cols

            select_cols = ["row_id", "row_index", "row_json"]
            if has_dataset_id:
                select_cols.append("dataset_version_id")
            select_cols.extend(
                [col for col in name_lookup.keys() if col in table_cols]
            )
            if not select_cols:
                return {
                    "dataset_version_id": dataset_version_id,
                    "table_name": table_name,
                    "rows": [],
                    "parameters": {},
                }

            scope = None
            if dataset_template and dataset_template.get("mapping_json"):
                try:
                    mapping = json.loads(dataset_template["mapping_json"])
                    scope = mapping.get("scope")
                except json.JSONDecodeError:
                    scope = None

            conditions = ["row_index = ?"]
            params: list[Any] = [int(row_index)]
            if has_dataset_id:
                if source_dataset_version_id:
                    conditions.append("dataset_version_id = ?")
                    params.append(source_dataset_version_id)
                elif scope != "all":
                    conditions.append("dataset_version_id = ?")
                    params.append(dataset_version_id)

            where = " AND ".join(conditions)
            sql = (
                f"SELECT {', '.join(quote_identifier(c) for c in select_cols)} "
                f"FROM {quote_identifier(table_name)} WHERE {where} "
                "ORDER BY row_index LIMIT ?"
            )
            params.append(int(max_rows))
            cur = conn.execute(sql, params)
            raw_rows = [dict(row) for row in cur.fetchall()]

            rows: list[dict[str, Any]] = []
            row_params: dict[str, list[dict[str, Any]]] = {}
            for row in raw_rows:
                values: dict[str, Any] = {}
                for safe_name, friendly in name_lookup.items():
                    if safe_name in row:
                        values[friendly] = row.get(safe_name)
                row_key = f"{row.get('dataset_version_id', dataset_version_id)}:{row.get('row_index')}"
                rows.append(
                    {
                        "row_id": row.get("row_id"),
                        "row_index": row.get("row_index"),
                        "dataset_version_id": row.get(
                            "dataset_version_id", dataset_version_id
                        ),
                        "row_json": row.get("row_json"),
                        "values": values,
                    }
                )
                row_params[row_key] = []

            if rows:
                by_dataset: dict[str, list[int]] = {}
                for row in rows:
                    key = row.get("dataset_version_id") or dataset_version_id
                    by_dataset.setdefault(key, []).append(int(row["row_index"]))
                for dv_id, indices in by_dataset.items():
                    placeholders = ", ".join(["?"] * len(indices))
                    params = [dv_id, *indices]
                    link_rows = conn.execute(
                        f"""
                        SELECT rpl.row_index, pe.entity_id, pe.canonical_text
                        FROM row_parameter_link rpl
                        JOIN parameter_entities pe ON pe.entity_id = rpl.entity_id
                        WHERE rpl.dataset_version_id = ? AND rpl.row_index IN ({placeholders})
                        """,
                        params,
                    ).fetchall()
                    entity_ids = sorted(
                        {int(row["entity_id"]) for row in link_rows}
                    )
                    kv_map: dict[int, list[dict[str, Any]]] = {}
                    if entity_ids:
                        kv_placeholders = ", ".join(["?"] * len(entity_ids))
                        kv_rows = conn.execute(
                            f"""
                            SELECT entity_id, key, value
                            FROM parameter_kv
                            WHERE entity_id IN ({kv_placeholders})
                            """,
                            entity_ids,
                        ).fetchall()
                        for kv in kv_rows:
                            kv_map.setdefault(int(kv["entity_id"]), []).append(
                                {"key": kv["key"], "value": kv["value"]}
                            )
                    for link in link_rows:
                        row_key = f"{dv_id}:{link['row_index']}"
                        row_params.setdefault(row_key, []).append(
                            {
                                "entity_id": int(link["entity_id"]),
                                "canonical": link["canonical_text"],
                                "kv": kv_map.get(int(link["entity_id"]), []),
                            }
                        )

            for row in rows:
                row_key = f"{row.get('dataset_version_id', dataset_version_id)}:{row.get('row_index')}"
                row["parameters"] = row_params.get(row_key, [])

            return {
                "dataset_version_id": dataset_version_id,
                "table_name": table_name,
                "rows": rows,
                "parameters": row_params,
            }

    def list_projects(self) -> list[dict[str, Any]]:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT p.project_id, p.fingerprint, p.name, p.created_at, p.erp_type,
                       (SELECT COUNT(*) FROM datasets d
                        WHERE d.project_id = p.project_id AND d.tenant_id = p.tenant_id) AS dataset_count
                FROM projects p
                WHERE p.tenant_id = ?
                ORDER BY p.created_at DESC
                """
                ,
                (tenant_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def list_runs_by_project(self, project_id: str, limit: int = 25) -> list[dict[str, Any]]:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT run_id, created_at, status, input_filename, upload_id, dataset_version_id
                FROM runs
                WHERE project_id = ? AND tenant_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (project_id, tenant_id, int(limit)),
            )
            return [dict(row) for row in cur.fetchall()]

    def fetch_project(self, project_id: str) -> dict[str, Any] | None:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                "SELECT project_id, fingerprint, name, created_at, erp_type FROM projects WHERE project_id = ? AND tenant_id = ?",
                (project_id, tenant_id),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def update_project_name(self, project_id: str, name: str) -> None:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            conn.execute(
                "UPDATE projects SET name = ? WHERE project_id = ? AND tenant_id = ?",
                (name, project_id, tenant_id),
            )

    def fetch_project_plugin_settings(self, project_id: str) -> dict[str, dict[str, Any]]:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT plugin_id, settings_json
                FROM project_plugin_settings
                WHERE project_id = ? AND tenant_id = ?
                """,
                (project_id, tenant_id),
            )
            settings: dict[str, dict[str, Any]] = {}
            for row in cur.fetchall():
                payload = {}
                if row["settings_json"]:
                    try:
                        payload = json.loads(row["settings_json"])
                    except json.JSONDecodeError:
                        payload = {}
                settings[str(row["plugin_id"])] = payload
            return settings

    def replace_project_plugin_settings(
        self, project_id: str, settings: dict[str, dict[str, Any]]
    ) -> None:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            conn.execute(
                "DELETE FROM project_plugin_settings WHERE project_id = ? AND tenant_id = ?",
                (project_id, tenant_id),
            )
            if not settings:
                return
            created_at = now_iso()
            rows = []
            for plugin_id, payload in settings.items():
                rows.append(
                    (
                        project_id,
                        tenant_id,
                        plugin_id,
                        json_dumps(payload) if payload is not None else None,
                        created_at,
                        created_at,
                    )
                )
            conn.executemany(
                """
                INSERT INTO project_plugin_settings
                (project_id, tenant_id, plugin_id, settings_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def update_project_erp_type(self, project_id: str, erp_type: str) -> None:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            conn.execute(
                "UPDATE projects SET erp_type = ? WHERE project_id = ? AND tenant_id = ?",
                (erp_type, project_id, tenant_id),
            )

    def fetch_project_role_overrides(self, project_id: str) -> dict[str, str]:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT role, column_name
                FROM project_role_overrides
                WHERE project_id = ? AND tenant_id = ?
                """,
                (project_id, tenant_id),
            )
            return {row["role"]: row["column_name"] for row in cur.fetchall()}

    def replace_project_role_overrides(
        self, project_id: str, overrides: dict[str, str]
    ) -> None:
        tenant_id = self._tenant_id()
        created_at = now_iso()
        with self.connection() as conn:
            conn.execute(
                "DELETE FROM project_role_overrides WHERE project_id = ? AND tenant_id = ?",
                (project_id, tenant_id),
            )
            rows = [
                (
                    project_id,
                    tenant_id,
                    role,
                    column_name,
                    created_at,
                    created_at,
                )
                for role, column_name in overrides.items()
                if column_name
            ]
            if rows:
                conn.executemany(
                    """
                    INSERT INTO project_role_overrides
                    (project_id, tenant_id, role, column_name, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

    def replace_dataset_role_candidates(
        self, dataset_version_id: str, candidates: list[dict[str, Any]]
    ) -> None:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            conn.execute(
                "DELETE FROM dataset_role_candidates WHERE dataset_version_id = ? AND tenant_id = ?",
                (dataset_version_id, tenant_id),
            )
            if not candidates:
                return
            created_at = now_iso()
            rows = [
                (
                    dataset_version_id,
                    tenant_id,
                    int(item["column_id"]),
                    item["role"],
                    float(item["score"]),
                    json_dumps(item.get("reasons") or []),
                    created_at,
                )
                for item in candidates
            ]
            conn.executemany(
                """
                INSERT INTO dataset_role_candidates
                (dataset_version_id, tenant_id, column_id, role, score, reasons_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def fetch_dataset_role_candidates(
        self, dataset_version_id: str
    ) -> list[dict[str, Any]]:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT column_id, role, score, reasons_json
                FROM dataset_role_candidates
                WHERE dataset_version_id = ? AND tenant_id = ?
                ORDER BY role, score DESC
                """,
                (dataset_version_id, tenant_id),
            )
            rows = []
            for row in cur.fetchall():
                entry = dict(row)
                if entry.get("reasons_json"):
                    try:
                        entry["reasons"] = json.loads(entry["reasons_json"])
                    except json.JSONDecodeError:
                        entry["reasons"] = []
                rows.append(entry)
            return rows

    def select_role_assignments(
        self, dataset_version_id: str, project_id: str | None = None
    ) -> dict[str, dict[str, Any]]:
        overrides = (
            self.fetch_project_role_overrides(project_id)
            if project_id
            else {}
        )
        columns = self.fetch_dataset_columns(dataset_version_id)
        by_name = {col["original_name"]: col for col in columns}
        assignments: dict[str, dict[str, Any]] = {}
        for role, column_name in overrides.items():
            col = by_name.get(column_name)
            if col:
                assignments[role] = col
        if assignments and len(assignments) == len(overrides):
            return assignments

        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT column_id, role, score
                FROM dataset_role_candidates
                WHERE dataset_version_id = ? AND tenant_id = ?
                ORDER BY role, score DESC
                """,
                (dataset_version_id, self._tenant_id()),
            )
            role_best: dict[str, tuple[int, float]] = {}
            for row in cur.fetchall():
                role = row["role"]
                if role in assignments:
                    continue
                if role not in role_best:
                    role_best[role] = (int(row["column_id"]), float(row["score"]))
        if role_best:
            columns_by_id = {col["column_id"]: col for col in columns}
            for role, (col_id, _) in role_best.items():
                col = columns_by_id.get(col_id)
                if col:
                    assignments[role] = col
        return assignments

    def list_dataset_versions_by_project(self, project_id: str) -> list[dict[str, Any]]:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT dv.dataset_version_id, dv.dataset_id, dv.created_at, dv.table_name,
                       dv.row_count, dv.column_count, dv.data_hash, dv.raw_format_id,
                       (SELECT r.input_filename FROM runs r
                        WHERE r.dataset_version_id = dv.dataset_version_id
                          AND r.tenant_id = ?
                        ORDER BY r.created_at ASC LIMIT 1) AS source_filename,
                       (SELECT r.created_at FROM runs r
                        WHERE r.dataset_version_id = dv.dataset_version_id
                          AND r.tenant_id = ?
                        ORDER BY r.created_at ASC LIMIT 1) AS first_run_at,
                       (SELECT r.created_at FROM runs r
                        WHERE r.dataset_version_id = dv.dataset_version_id
                          AND r.tenant_id = ?
                        ORDER BY r.created_at DESC LIMIT 1) AS last_run_at
                FROM dataset_versions dv
                JOIN datasets d ON d.dataset_id = dv.dataset_id
                WHERE d.project_id = ? AND dv.tenant_id = ? AND d.tenant_id = ?
                ORDER BY dv.created_at DESC
                """,
                (tenant_id, tenant_id, tenant_id, project_id, tenant_id, tenant_id),
            )
            return [dict(row) for row in cur.fetchall()]

    def list_dataset_versions(self) -> list[dict[str, Any]]:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT dataset_version_id, dataset_id, created_at, table_name, row_count,
                       column_count, data_hash, raw_format_id
                FROM dataset_versions
                WHERE tenant_id = ?
                ORDER BY created_at DESC
                """,
                (tenant_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def fetch_latest_plugin_results_for_dataset(
        self, dataset_version_id: str
    ) -> list[dict[str, Any]]:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT pr.*
                FROM plugin_results_v2 pr
                JOIN runs r ON r.run_id = pr.run_id
                JOIN (
                    SELECT pr2.plugin_id AS plugin_id, MAX(pr2.result_id) AS max_id
                    FROM plugin_results_v2 pr2
                    JOIN runs r2 ON r2.run_id = pr2.run_id
                    WHERE r2.dataset_version_id = ? AND r2.tenant_id = ? AND pr2.tenant_id = ?
                    GROUP BY pr2.plugin_id
                ) latest
                ON pr.plugin_id = latest.plugin_id AND pr.result_id = latest.max_id
                WHERE r.dataset_version_id = ? AND r.tenant_id = ? AND pr.tenant_id = ?
                """,
                (dataset_version_id, tenant_id, tenant_id, dataset_version_id, tenant_id, tenant_id),
            )
            return [dict(row) for row in cur.fetchall()]

    def fetch_deliveries_for_dataset(
        self, dataset_version_id: str
    ) -> list[dict[str, Any]]:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT *
                FROM deliveries
                WHERE dataset_version_id = ? AND tenant_id = ?
                ORDER BY delivered_at DESC
                """,
                (dataset_version_id, tenant_id),
            )
            return [dict(row) for row in cur.fetchall()]

    def record_delivery(
        self,
        project_id: str,
        dataset_version_id: str,
        plugin_id: str,
        plugin_version: str | None,
        code_hash: str | None,
        dataset_hash: str | None,
        delivered_at: str,
        notes: str | None = None,
    ) -> None:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO deliveries
                (project_id, tenant_id, dataset_version_id, plugin_id, plugin_version, code_hash, dataset_hash, delivered_at, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    tenant_id,
                    dataset_version_id,
                    plugin_id,
                    plugin_version,
                    code_hash,
                    dataset_hash,
                    delivered_at,
                    notes,
                ),
            )

    def enqueue_analysis_job(
        self,
        dataset_version_id: str,
        plugin_id: str,
        plugin_version: str | None,
        code_hash: str | None,
        settings_hash: str | None,
        run_seed: int,
        created_at: str,
    ) -> None:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO analysis_jobs
                (dataset_version_id, tenant_id, plugin_id, plugin_version, code_hash, settings_hash, run_seed, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?)
                """,
                (
                    dataset_version_id,
                    tenant_id,
                    plugin_id,
                    plugin_version,
                    code_hash,
                    settings_hash,
                    int(run_seed),
                    created_at,
                ),
            )

    def list_analysis_jobs(self, status: str = "queued") -> list[dict[str, Any]]:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                "SELECT * FROM analysis_jobs WHERE status = ? AND tenant_id = ? ORDER BY created_at",
                (status, tenant_id),
            )
            return [dict(row) for row in cur.fetchall()]

    def update_analysis_job_status(
        self,
        job_id: int,
        status: str,
        started_at: str | None = None,
        completed_at: str | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE analysis_jobs
                SET status = ?, started_at = COALESCE(?, started_at),
                    completed_at = COALESCE(?, completed_at),
                    error_json = ?
                WHERE job_id = ? AND tenant_id = ?
                """,
                (
                    status,
                    started_at,
                    completed_at,
                    json_dumps(error) if error else None,
                    job_id,
                    tenant_id,
                ),
            )

    def create_dataset_table(
        self,
        table_name: str,
        columns: list[dict[str, Any]],
        conn: sqlite3.Connection | None = None,
    ) -> None:
        if conn is None:
            with self.connection() as temp:
                self.create_dataset_table(table_name, columns, temp)
                return
        safe_table = quote_identifier(table_name)
        col_defs = []
        for col in columns:
            safe_name = quote_identifier(col["safe_name"])
            col_type = col.get("sqlite_type", "TEXT")
            col_defs.append(f"{safe_name} {col_type}")
        ddl = (
            f"CREATE TABLE IF NOT EXISTS {safe_table} ("
            "row_id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "row_index INTEGER NOT NULL, "
            "row_json TEXT, "
            + ", ".join(col_defs)
            + ")"
        )
        conn.execute(ddl)

    def create_template_table(
        self,
        table_name: str,
        fields: list[dict[str, Any]],
        conn: sqlite3.Connection | None = None,
    ) -> None:
        if conn is None:
            with self.connection() as temp:
                self.create_template_table(table_name, fields, temp)
                return
        safe_table = quote_identifier(table_name)
        col_defs = []
        for field in fields:
            safe_name = quote_identifier(field["safe_name"])
            col_type = field.get("sqlite_type", "TEXT")
            col_defs.append(f"{safe_name} {col_type}")
        ddl = (
            f"CREATE TABLE IF NOT EXISTS {safe_table} ("
            "row_id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "dataset_version_id TEXT NOT NULL, "
            "row_index INTEGER NOT NULL, "
            "row_json TEXT, "
            + ", ".join(col_defs)
            + ")"
        )
        conn.execute(ddl)
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_dataset ON {safe_table}(dataset_version_id)"
        )

    def add_append_only_triggers(
        self, table_name: str, conn: sqlite3.Connection | None = None
    ) -> None:
        if conn is None:
            with self.connection() as temp:
                self.add_append_only_triggers(table_name, temp)
                return
        safe_table = quote_identifier(table_name)
        conn.execute(
            f"""
            CREATE TRIGGER IF NOT EXISTS {table_name}_no_delete
            BEFORE DELETE ON {safe_table}
            BEGIN
                SELECT RAISE(ABORT, 'raw data is append-only');
            END;
            """
        )
        conn.execute(
            f"""
            CREATE TRIGGER IF NOT EXISTS {table_name}_no_update
            BEFORE UPDATE ON {safe_table}
            BEGIN
                SELECT RAISE(ABORT, 'raw data is append-only');
            END;
            """
        )

    def insert_dataset_rows(
        self,
        table_name: str,
        safe_columns: list[str],
        rows: list[tuple[Any, ...]],
        conn: sqlite3.Connection | None = None,
    ) -> None:
        if not rows:
            return
        if conn is None:
            with self.connection() as temp:
                self.insert_dataset_rows(table_name, safe_columns, rows, temp)
                return
        quoted_cols = [quote_identifier(col) for col in safe_columns]
        cols = ["row_index", "row_json"] + quoted_cols
        placeholders = ", ".join(["?"] * len(cols))
        sql = (
            f"INSERT INTO {quote_identifier(table_name)} "
            f"({', '.join(cols)}) VALUES ({placeholders})"
        )
        conn.executemany(sql, rows)

    def insert_template_rows(
        self,
        table_name: str,
        safe_columns: list[str],
        rows: list[tuple[Any, ...]],
        conn: sqlite3.Connection | None = None,
    ) -> None:
        if not rows:
            return
        if conn is None:
            with self.connection() as temp:
                self.insert_template_rows(table_name, safe_columns, rows, temp)
                return
        quoted_cols = [quote_identifier(col) for col in safe_columns]
        cols = ["dataset_version_id", "row_index", "row_json"] + quoted_cols
        placeholders = ", ".join(["?"] * len(cols))
        sql = (
            f"INSERT INTO {quote_identifier(table_name)} "
            f"({', '.join(cols)}) VALUES ({placeholders})"
        )
        conn.executemany(sql, rows)

    def close(self) -> None:
        return

    def count_users(self) -> int:
        with self.connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM users")
            return int(cur.fetchone()[0])

    def list_tenants(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            cur = conn.execute(
                "SELECT tenant_id, name, created_at, is_default FROM tenants ORDER BY created_at"
            )
            return [dict(row) for row in cur.fetchall()]

    def create_tenant(self, tenant_id: str, name: str | None, created_at: str) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO tenants (tenant_id, name, created_at, is_default)
                VALUES (?, ?, ?, 0)
                """,
                (tenant_id, name, created_at),
            )

    def create_user(
        self,
        email: str,
        password_hash: str,
        name: str | None,
        is_admin: bool,
        created_at: str,
    ) -> int:
        normalized = email.strip().lower()
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO users (email, name, password_hash, is_admin, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (normalized, name, password_hash, 1 if is_admin else 0, created_at),
            )
            return int(cur.lastrowid)

    def fetch_user_by_email(self, email: str) -> dict[str, Any] | None:
        normalized = email.strip().lower()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT user_id, email, name, password_hash, is_admin, created_at, disabled_at
                FROM users
                WHERE email = ?
                """,
                (normalized,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def fetch_user_by_id(self, user_id: int) -> dict[str, Any] | None:
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT user_id, email, name, password_hash, is_admin, created_at, disabled_at
                FROM users
                WHERE user_id = ?
                """,
                (int(user_id),),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def ensure_membership(
        self, user_id: int, role: str, created_at: str, tenant_id: str | None = None
    ) -> None:
        tenant_id = tenant_id or self._tenant_id()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO tenant_memberships
                (tenant_id, user_id, role, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (tenant_id, int(user_id), role, created_at),
            )

    def fetch_membership(
        self, user_id: int, tenant_id: str | None = None
    ) -> dict[str, Any] | None:
        tenant_id = tenant_id or self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT membership_id, tenant_id, user_id, role, created_at
                FROM tenant_memberships
                WHERE tenant_id = ? AND user_id = ?
                """,
                (tenant_id, int(user_id)),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def list_users_for_tenant(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
        tenant_id = tenant_id or self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT u.user_id, u.email, u.name, u.is_admin, u.created_at, u.disabled_at,
                       m.role
                FROM tenant_memberships m
                JOIN users u ON u.user_id = m.user_id
                WHERE m.tenant_id = ?
                ORDER BY u.created_at
                """,
                (tenant_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def create_session(
        self,
        user_id: int,
        token_hash: str,
        created_at: str,
        expires_at: str,
        tenant_id: str | None = None,
    ) -> int:
        tenant_id = tenant_id or self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO user_sessions
                (user_id, tenant_id, token_hash, created_at, last_seen_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    int(user_id),
                    tenant_id,
                    token_hash,
                    created_at,
                    created_at,
                    expires_at,
                ),
            )
            return int(cur.lastrowid)

    def fetch_session_by_hash(
        self, token_hash: str, tenant_id: str | None = None
    ) -> dict[str, Any] | None:
        tenant_id = tenant_id or self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT s.session_id, s.user_id, s.tenant_id, s.token_hash, s.created_at,
                       s.last_seen_at, s.expires_at, s.revoked_at,
                       u.email, u.name, u.is_admin, u.disabled_at
                FROM user_sessions s
                JOIN users u ON u.user_id = s.user_id
                WHERE s.token_hash = ? AND s.tenant_id = ?
                """,
                (token_hash, tenant_id),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def touch_session(self, session_id: int, seen_at: str) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE user_sessions SET last_seen_at = ? WHERE session_id = ?",
                (seen_at, int(session_id)),
            )

    def rotate_session(
        self, session_id: int, token_hash: str, seen_at: str, expires_at: str
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE user_sessions
                SET token_hash = ?, last_seen_at = ?, expires_at = ?, revoked_at = NULL
                WHERE session_id = ?
                """,
                (token_hash, seen_at, expires_at, int(session_id)),
            )

    def revoke_session(self, token_hash: str) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE user_sessions SET revoked_at = ? WHERE token_hash = ?",
                (now_iso(), token_hash),
            )

    def create_api_key(
        self,
        user_id: int,
        key_hash: str,
        name: str | None,
        created_at: str,
        tenant_id: str | None = None,
    ) -> int:
        tenant_id = tenant_id or self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO api_keys
                (user_id, tenant_id, name, key_hash, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (int(user_id), tenant_id, name, key_hash, created_at),
            )
            return int(cur.lastrowid)

    def fetch_api_key_by_hash(
        self, key_hash: str, tenant_id: str | None = None
    ) -> dict[str, Any] | None:
        tenant_id = tenant_id or self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT k.key_id, k.user_id, k.tenant_id, k.name, k.key_hash, k.created_at,
                       k.last_used_at, k.revoked_at,
                       u.email, u.name AS user_name, u.is_admin, u.disabled_at
                FROM api_keys k
                JOIN users u ON u.user_id = k.user_id
                WHERE k.key_hash = ? AND k.tenant_id = ?
                """,
                (key_hash, tenant_id),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def list_api_keys(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
        tenant_id = tenant_id or self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT k.key_id, k.user_id, k.tenant_id, k.name, k.created_at,
                       k.last_used_at, k.revoked_at, u.email
                FROM api_keys k
                JOIN users u ON u.user_id = k.user_id
                WHERE k.tenant_id = ?
                ORDER BY k.created_at DESC
                """,
                (tenant_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def touch_api_key(self, key_id: int, seen_at: str) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE key_id = ?",
                (seen_at, int(key_id)),
            )

    def revoke_api_key(self, key_id: int) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE key_id = ?",
                (now_iso(), int(key_id)),
            )

    def disable_user(self, user_id: int) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE users SET disabled_at = ? WHERE user_id = ?",
                (now_iso(), int(user_id)),
            )

    def revoke_user_sessions(self, user_id: int) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE user_sessions SET revoked_at = ? WHERE user_id = ?",
                (now_iso(), int(user_id)),
            )

    def revoke_user_api_keys(self, user_id: int) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE user_id = ?",
                (now_iso(), int(user_id)),
            )

    def ensure_raw_format(
        self, fingerprint: str, name: str | None = None, created_at: str | None = None
    ) -> int:
        tenant_id = self._tenant_id()
        scoped_fingerprint = self._scoped_value(fingerprint)
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO raw_formats (fingerprint, tenant_id, name, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (scoped_fingerprint, tenant_id, name, created_at),
            )
            cur = conn.execute(
                "SELECT format_id FROM raw_formats WHERE fingerprint = ? AND tenant_id = ?",
                (scoped_fingerprint, tenant_id),
            )
            row = cur.fetchone()
            if not row:
                raise ValueError("Raw format not found")
            return int(row[0])

    def list_raw_formats(self) -> list[dict[str, Any]]:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT rf.format_id, rf.fingerprint, rf.name, rf.created_at,
                       (SELECT COUNT(*) FROM dataset_versions dv
                        WHERE dv.raw_format_id = rf.format_id AND dv.tenant_id = rf.tenant_id) AS dataset_count
                FROM raw_formats rf
                WHERE rf.tenant_id = ?
                ORDER BY rf.created_at DESC
                """,
                (tenant_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def add_raw_format_mapping(
        self,
        format_id: int,
        template_id: int,
        mapping_json: str,
        mapping_hash: str,
        notes: str | None,
        created_at: str,
    ) -> None:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO raw_format_mappings
                (format_id, tenant_id, template_id, mapping_json, mapping_hash, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (format_id, tenant_id, template_id, mapping_json, mapping_hash, notes, created_at),
            )

    def list_raw_format_mappings(self, format_id: int) -> list[dict[str, Any]]:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT mapping_id, template_id, mapping_json, mapping_hash, notes, created_at
                FROM raw_format_mappings
                WHERE format_id = ? AND tenant_id = ?
                ORDER BY created_at DESC
                """,
                (format_id, tenant_id),
            )
            return [dict(row) for row in cur.fetchall()]

    def add_raw_format_note(
        self, format_id: int, note: str, created_at: str
    ) -> None:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO raw_format_notes (format_id, tenant_id, note, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (format_id, tenant_id, note, created_at),
            )

    def list_raw_format_notes(self, format_id: int) -> list[dict[str, Any]]:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT note_id, note, created_at
                FROM raw_format_notes
                WHERE format_id = ? AND tenant_id = ?
                ORDER BY created_at DESC
                """,
                (format_id, tenant_id),
            )
            return [dict(row) for row in cur.fetchall()]

    def create_template(
        self,
        name: str,
        fields: list[dict[str, Any]],
        description: str | None,
        version: str | None,
        created_at: str,
    ) -> int:
        tenant_id = self._tenant_id()
        scoped_name = self._scoped_value(name)
        table_name = f"template_{scoped_name.lower().replace(' ', '_').replace(':', '_')}"
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO templates (name, tenant_id, description, version, created_at, table_name)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (scoped_name, tenant_id, description, version, created_at, table_name),
            )
            cur = conn.execute(
                "SELECT template_id FROM templates WHERE name = ? AND tenant_id = ?",
                (scoped_name, tenant_id),
            )
            row = cur.fetchone()
            if not row:
                raise ValueError("Template not created")
            template_id = int(row[0])
            prepared_fields = []
            for idx, field in enumerate(fields, start=1):
                prepared_fields.append(
                    {
                        "field_id": idx,
                        "safe_name": f"f{idx}",
                        "name": field["name"],
                        "dtype": field.get("dtype"),
                        "role": field.get("role"),
                        "required": 1 if field.get("required") else 0,
                        "sqlite_type": field.get("sqlite_type", "TEXT"),
                    }
                )
            conn.executemany(
                """
                INSERT INTO template_fields
                (template_id, tenant_id, field_id, safe_name, name, dtype, role, required)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        template_id,
                        tenant_id,
                        f["field_id"],
                        f["safe_name"],
                        f["name"],
                        f.get("dtype"),
                        f.get("role"),
                        f.get("required", 0),
                    )
                    for f in prepared_fields
                ],
            )
            self.create_template_table(table_name, prepared_fields, conn)
        return template_id

    def list_templates(self) -> list[dict[str, Any]]:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                "SELECT * FROM templates WHERE tenant_id = ? ORDER BY created_at DESC",
                (tenant_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def fetch_template(self, template_id: int) -> dict[str, Any] | None:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                "SELECT * FROM templates WHERE template_id = ? AND tenant_id = ?",
                (template_id, tenant_id),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def fetch_template_fields(self, template_id: int) -> list[dict[str, Any]]:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT field_id, safe_name, name, dtype, role, required
                FROM template_fields
                WHERE template_id = ? AND tenant_id = ?
                ORDER BY field_id
                """,
                (template_id, tenant_id),
            )
            return [dict(row) for row in cur.fetchall()]

    def upsert_dataset_template(
        self,
        dataset_version_id: str,
        template_id: int,
        mapping_json: str,
        mapping_hash: str,
        status: str,
        created_at: str,
        updated_at: str,
    ) -> None:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO dataset_templates
                (dataset_version_id, tenant_id, template_id, mapping_json, mapping_hash, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_version_id, template_id)
                DO UPDATE SET mapping_json = excluded.mapping_json,
                              mapping_hash = excluded.mapping_hash,
                              status = excluded.status,
                              updated_at = excluded.updated_at
                """,
                (
                    dataset_version_id,
                    tenant_id,
                    template_id,
                    mapping_json,
                    mapping_hash,
                    status,
                    created_at,
                    updated_at,
                ),
            )

    def fetch_dataset_template(
        self, dataset_version_id: str
    ) -> dict[str, Any] | None:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT dt.*, t.table_name, t.name AS template_name, t.version AS template_version
                FROM dataset_templates dt
                JOIN templates t ON t.template_id = dt.template_id
                WHERE dt.dataset_version_id = ? AND dt.tenant_id = ? AND t.tenant_id = ?
                ORDER BY dt.updated_at DESC
                LIMIT 1
                """,
                (dataset_version_id, tenant_id, tenant_id),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def ensure_template_aggregate_dataset(
        self,
        template_id: int,
        created_at: str,
        filters: dict[str, Any] | None = None,
    ) -> str:
        template = self.fetch_template(template_id)
        if not template:
            raise ValueError("Template not found")
        mapping: dict[str, Any] = {"scope": "all"}
        if filters:
            mapping["filters"] = filters
        mapping_hash = hashlib.sha256(json_dumps(mapping).encode("utf-8")).hexdigest()
        if filters:
            dataset_version_id = f"template_{template_id}_all_{mapping_hash[:8]}"
        else:
            dataset_version_id = f"template_{template_id}_all"
        project_id = f"template_{template_id}_meta"
        fingerprint = dataset_version_id
        self.ensure_project(project_id, fingerprint, created_at)
        self.ensure_dataset(dataset_version_id, project_id, fingerprint, created_at)
        self.ensure_dataset_version(
            dataset_version_id,
            dataset_version_id,
            created_at,
            template["table_name"],
            f"template:{template_id}:all:{mapping_hash}",
        )
        self.upsert_dataset_template(
            dataset_version_id,
            template_id,
            json_dumps(mapping),
            mapping_hash,
            "ready",
            created_at,
            created_at,
        )
        return dataset_version_id

    def record_template_conversion(
        self,
        dataset_version_id: str,
        template_id: int,
        status: str,
        started_at: str | None,
        completed_at: str | None,
        mapping_hash: str,
        row_count: int = 0,
        error: dict[str, Any] | None = None,
    ) -> None:
        tenant_id = self._tenant_id()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO template_conversions
                (dataset_version_id, tenant_id, template_id, status, started_at, completed_at, error_json, mapping_hash, row_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_version_id,
                    tenant_id,
                    template_id,
                    status,
                    started_at,
                    completed_at,
                    json_dumps(error) if error else None,
                    mapping_hash,
                    int(row_count),
                ),
            )

    def trace_from_entity(
        self, entity_type: str, key: str, max_depth: int = 5
    ) -> dict[str, Any]:
        with self.connection() as conn:
            cur = conn.execute(
                """
                WITH RECURSIVE trace(entity_id, type, key, depth) AS (
                    SELECT entity_id, type, key, 0
                    FROM entities
                    WHERE type = ? AND key = ?
                    UNION ALL
                    SELECT e.dst_entity_id, ent.type, ent.key, trace.depth + 1
                    FROM edges e
                    JOIN trace ON e.src_entity_id = trace.entity_id
                    JOIN entities ent ON ent.entity_id = e.dst_entity_id
                    WHERE trace.depth < ?
                )
                SELECT entity_id, type, key, depth FROM trace
                """,
                (entity_type, key, max_depth),
            )
            nodes = [dict(row) for row in cur.fetchall()]
            node_ids = [row["entity_id"] for row in nodes]
            if not node_ids:
                return {"nodes": [], "edges": []}
            placeholders = ", ".join(["?"] * len(node_ids))
            edges_cur = conn.execute(
                f"""
                SELECT src_entity_id, dst_entity_id, kind, evidence_json, score
                FROM edges
                WHERE src_entity_id IN ({placeholders})
                """,
                node_ids,
            )
            return {"nodes": nodes, "edges": [dict(row) for row in edges_cur.fetchall()]}
