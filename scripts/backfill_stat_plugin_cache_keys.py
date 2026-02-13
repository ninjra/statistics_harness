from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from statistic_harness.core.stat_plugins.code_hash import stat_plugin_effective_code_hash
from statistic_harness.core.utils import file_sha256, json_dumps


REPO_ROOT = Path(__file__).resolve().parents[1]


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _latest_completed_run(conn: sqlite3.Connection, dataset_version_id: str) -> str | None:
    row = conn.execute(
        """
        SELECT run_id
        FROM runs
        WHERE status = 'completed' AND dataset_version_id = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (dataset_version_id,),
    ).fetchone()
    return str(row["run_id"]) if row else None


def _execution_fingerprint(
    *, plugin_id: str, plugin_version: str | None, code_hash: str | None, settings_hash: str | None, dataset_hash: str | None
) -> str:
    payload = {
        "plugin_id": plugin_id,
        "plugin_version": plugin_version,
        "code_hash": code_hash,
        "settings_hash": settings_hash,
        "dataset_hash": dataset_hash,
    }
    return hashlib.sha256(json_dumps(payload).encode("utf-8")).hexdigest()


def _is_stat_wrapper(module_file: Path) -> bool:
    try:
        text = module_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return "run_plugin(" in text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="appdata/state.sqlite")
    ap.add_argument("--dataset-version-id", default="")
    ap.add_argument("--source-run-id", default="")
    ap.add_argument("--exclude-plugin-id", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db_path = (REPO_ROOT / str(args.db)).resolve()
    if not db_path.exists():
        raise SystemExit(f"Missing DB: {db_path}")

    excluded = {str(x).strip() for x in (args.exclude_plugin_id or []) if str(x).strip()}

    conn = _connect(db_path)
    try:
        dataset_version_id = str(args.dataset_version_id or "").strip()
        if not dataset_version_id:
            row = conn.execute(
                "SELECT dataset_version_id FROM dataset_versions ORDER BY row_count DESC, created_at DESC LIMIT 1"
            ).fetchone()
            if not row:
                raise SystemExit("No dataset_versions found")
            dataset_version_id = str(row["dataset_version_id"])

        source_run_id = str(args.source_run_id or "").strip()
        if not source_run_id:
            source_run_id = _latest_completed_run(conn, dataset_version_id) or ""
        if not source_run_id:
            raise SystemExit(f"No completed run found for dataset_version_id={dataset_version_id}")

        rows = conn.execute(
            """
            SELECT *
            FROM plugin_results_v2
            WHERE run_id = ? AND tenant_id = 'default'
            """,
            (source_run_id,),
        ).fetchall()

        inserted = 0
        considered = 0
        skipped = 0
        for row in rows:
            plugin_id = str(row["plugin_id"] or "")
            if not plugin_id or plugin_id in excluded:
                skipped += 1
                continue
            status = str(row["status"] or "")
            if status != "ok":
                skipped += 1
                continue
            module_file = REPO_ROOT / "plugins" / plugin_id / "plugin.py"
            if not module_file.exists() or not _is_stat_wrapper(module_file):
                skipped += 1
                continue
            wrapper_hash = file_sha256(module_file)
            effective = stat_plugin_effective_code_hash(plugin_id)
            if not isinstance(effective, str) or not effective:
                skipped += 1
                continue
            new_code_hash = hashlib.sha256(f"{wrapper_hash}:{effective}".encode("utf-8")).hexdigest()
            new_fp = _execution_fingerprint(
                plugin_id=plugin_id,
                plugin_version=row["plugin_version"],
                code_hash=new_code_hash,
                settings_hash=row["settings_hash"],
                dataset_hash=row["dataset_hash"],
            )
            considered += 1
            exists = conn.execute(
                "SELECT 1 FROM plugin_results_v2 WHERE tenant_id='default' AND execution_fingerprint=? LIMIT 1",
                (new_fp,),
            ).fetchone()
            if exists:
                skipped += 1
                continue

            if args.dry_run:
                inserted += 1
                continue

            conn.execute(
                """
                INSERT INTO plugin_results_v2
                (run_id, plugin_id, plugin_version, executed_at, code_hash, settings_hash, dataset_hash,
                 status, summary, metrics_json, findings_json, artifacts_json, error_json, budget_json,
                 tenant_id, references_json, debug_json, execution_fingerprint)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_run_id,
                    plugin_id,
                    row["plugin_version"],
                    row["executed_at"],
                    new_code_hash,
                    row["settings_hash"],
                    row["dataset_hash"],
                    row["status"],
                    row["summary"],
                    row["metrics_json"],
                    row["findings_json"],
                    row["artifacts_json"],
                    row["error_json"],
                    row["budget_json"],
                    row["tenant_id"],
                    row["references_json"],
                    row["debug_json"],
                    new_fp,
                ),
            )
            inserted += 1

        if not args.dry_run:
            conn.commit()

        result = {
            "dataset_version_id": dataset_version_id,
            "source_run_id": source_run_id,
            "excluded_plugin_ids": sorted(excluded),
            "considered": considered,
            "inserted": inserted,
            "skipped": skipped,
            "dry_run": bool(args.dry_run),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

