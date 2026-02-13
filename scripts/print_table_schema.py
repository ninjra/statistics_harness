#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path("appdata/state.sqlite"))
    ap.add_argument("--table", required=True)
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(f"pragma table_info({args.table})").fetchall()
        for r in rows:
            print(f"{r['name']}\t{r['type']}")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

