#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from statistic_harness.core.storage import Storage
from statistic_harness.core.utils import now_iso, quote_identifier


SYSTEM_COLS = {"row_id", "row_index", "row_json"}


def _raw_columns(conn: sqlite3.Connection, raw_table: str) -> list[dict[str, str]]:
    rows = conn.execute(f"PRAGMA table_info({quote_identifier(raw_table)})").fetchall()
    cols: list[dict[str, str]] = []
    for r in rows:
        name = str(r[1])
        if name in SYSTEM_COLS:
            continue
        col_type = str(r[2] or "")
        cols.append({"safe_name": name, "sqlite_type": col_type})
    return cols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path("appdata/state.sqlite"))
    ap.add_argument("--dataset-version-id", required=True)
    ap.add_argument("--raw-table-name", default="")
    ap.add_argument(
        "--project-id",
        default="",
        help="Project id to attach this dataset to. Default: dataset_version_id (safe for single-user recovery).",
    )
    ap.add_argument("--dataset-id", default="")
    ap.add_argument("--created-at", default="")
    args = ap.parse_args()

    db_path: Path = args.db
    dataset_version_id = str(args.dataset_version_id).strip()
    if not dataset_version_id:
        raise SystemExit("--dataset-version-id required")

    raw_table = str(args.raw_table_name).strip() or f"dataset_{dataset_version_id}"
    project_id = str(args.project_id).strip() or dataset_version_id
    dataset_id = str(args.dataset_id).strip() or dataset_version_id
    created_at = str(args.created_at).strip() or now_iso()

    storage = Storage(db_path, tenant_id=None)

    # Introspect raw table and row_count.
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        # Verify the raw table exists before writing metadata.
        trow = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (raw_table,),
        ).fetchone()
        if not trow:
            raise SystemExit(f"Raw table not found: {raw_table}")

        cols = _raw_columns(con, raw_table)
        row_count = int(
            con.execute(f"SELECT COUNT(*) AS c FROM {quote_identifier(raw_table)}").fetchone()["c"]
        )
    finally:
        con.close()

    # Create minimal project/dataset/version rows (idempotent).
    dataset_fingerprint = dataset_version_id
    project_fingerprint = f"project:{project_id}"
    storage.ensure_project(project_id, project_fingerprint, created_at)
    storage.ensure_dataset(dataset_id, project_id, dataset_fingerprint, created_at)
    storage.ensure_dataset_version(
        dataset_version_id=dataset_version_id,
        dataset_id=dataset_id,
        created_at=created_at,
        table_name=raw_table,
        data_hash=dataset_version_id,
    )

    # Restore dataset_columns with safe_name==original_name. This is the safest available
    # recovery without the original ingest manifest.
    columns_meta = []
    for idx, col in enumerate(cols):
        safe = col["safe_name"]
        columns_meta.append(
            {
                "column_id": int(idx),
                "safe_name": safe,
                "original_name": safe,
                "dtype": col.get("sqlite_type") or None,
                "role": None,
                "pii_tags": None,
                "stats": None,
            }
        )
    storage.replace_dataset_columns(dataset_version_id, columns_meta)
    storage.update_dataset_version_stats(dataset_version_id, row_count, len(columns_meta))

    # Restore performance-critical row_index index.
    with storage.connection() as conn2:
        storage.ensure_dataset_row_index_index(raw_table, conn2)
        try:
            conn2.execute("ANALYZE")
        except Exception:
            pass

    print(f"OK repaired dataset_version_id={dataset_version_id} raw_table={raw_table} rows={row_count} cols={len(columns_meta)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
