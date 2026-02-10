#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def _connect(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    return con


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path("appdata/state.sqlite"))
    ap.add_argument("--dataset-version-id", required=True)
    args = ap.parse_args()

    con = _connect(args.db)
    try:
        dv = con.execute(
            "select dataset_version_id, table_name, row_count, column_count from dataset_versions where dataset_version_id=?",
            (args.dataset_version_id,),
        ).fetchone()
        if dv is None:
            raise SystemExit(f"dataset_version_id not found: {args.dataset_version_id}")
        table_name = str(dv["table_name"])

        idx_rows = con.execute(f"pragma index_list({json.dumps(table_name)})").fetchall()
        indexes: list[dict[str, Any]] = []
        has_row_index_index = False
        for r in idx_rows:
            name = str(r["name"])
            cols = [c["name"] for c in con.execute(f"pragma index_info({json.dumps(name)})").fetchall()]
            indexes.append({"name": name, "unique": int(r["unique"]), "origin": r["origin"], "cols": cols})
            if cols == ["row_index"] or "row_index" in cols:
                has_row_index_index = True

        payload = {
            "dataset_version_id": str(dv["dataset_version_id"]),
            "table_name": table_name,
            "row_count": int(dv["row_count"] or 0),
            "column_count": int(dv["column_count"] or 0),
            "indexes": indexes,
            "has_row_index_index": bool(has_row_index_index),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

