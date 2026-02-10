#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import json


REPO_ROOT = Path(__file__).resolve().parents[1]
APPDATA = REPO_ROOT / "appdata"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--dest-root", default=str(APPDATA / "sqlpacks"))
    ap.add_argument(
        "--pack-relpath",
        default="artifacts/llm_text2sql_local_generate_v1/sql_pack.json",
    )
    args = ap.parse_args()

    run_id = str(args.run_id).strip()
    run_dir = APPDATA / "runs" / run_id
    pack_path = run_dir / str(args.pack_relpath)
    if not pack_path.exists():
        raise SystemExit(f"Missing pack: {pack_path}")

    payload = json.loads(pack_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("schema_hash"):
        raise SystemExit("Invalid sql_pack.json (missing schema_hash)")
    schema_hash = str(payload["schema_hash"])

    dest_root = Path(str(args.dest_root))
    dest_dir = dest_root / schema_hash
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "sql_pack.json"
    shutil.copy2(pack_path, dest)
    print(str(dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

