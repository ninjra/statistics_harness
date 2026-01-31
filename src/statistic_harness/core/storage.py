from __future__ import annotations

import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .types import PluginArtifact, PluginError, PluginResult
from .utils import ensure_dir, json_dumps


class Storage:
    def __init__(self, db_path: Path) -> None:
        ensure_dir(db_path.parent)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT,
                status TEXT,
                upload_id TEXT,
                input_filename TEXT,
                canonical_path TEXT,
                settings_json TEXT,
                error_json TEXT
            )
            """
        )
        cur.execute(
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
        self.conn.commit()

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
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO runs
            (run_id, created_at, status, upload_id, input_filename, canonical_path, settings_json, error_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                created_at,
                status,
                upload_id,
                input_filename,
                canonical_path,
                json_dumps(settings),
                json_dumps(error) if error else None,
            ),
        )
        self.conn.commit()

    def update_run_status(self, run_id: str, status: str, error: dict[str, Any] | None = None) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE runs SET status = ?, error_json = ? WHERE run_id = ?",
            (status, json_dumps(error) if error else None, run_id),
        )
        self.conn.commit()

    def save_plugin_result(self, run_id: str, plugin_id: str, result: PluginResult) -> None:
        cur = self.conn.cursor()
        artifacts = [asdict(a) for a in result.artifacts]
        error_payload = asdict(result.error) if result.error else None
        cur.execute(
            """
            INSERT OR REPLACE INTO plugin_results
            (run_id, plugin_id, status, summary, metrics_json, findings_json, artifacts_json, error_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                plugin_id,
                result.status,
                result.summary,
                json_dumps(result.metrics),
                json_dumps(result.findings),
                json_dumps(artifacts),
                json_dumps(error_payload) if error_payload else None,
            ),
        )
        self.conn.commit()

    def fetch_plugin_results(self, run_id: str) -> list[dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM plugin_results WHERE run_id = ?", (run_id,))
        return [dict(row) for row in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()
