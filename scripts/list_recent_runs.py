#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path("appdata/state.sqlite"))
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--dataset-version-id", default="")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    try:
        cols = [r["name"] for r in con.execute("pragma table_info(runs)").fetchall()]
        want = ["run_id", "status", "created_at", "completed_at", "dataset_version_id"]
        have = [c for c in want if c in cols]
        if not have:
            raise SystemExit("runs table missing expected columns")
        sql = "select " + ", ".join(have) + " from runs"
        params: tuple[object, ...] = ()
        if args.dataset_version_id:
            sql += " where dataset_version_id=?"
            params = (args.dataset_version_id,)
        sql += " order by created_at desc limit ?"
        params = (*params, int(args.limit))
        rows = con.execute(sql, params).fetchall()
        for r in rows:
            parts = [str(r[c]) for c in have]
            print("\t".join(parts))
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
