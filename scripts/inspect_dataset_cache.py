#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from statistic_harness.core.dataset_cache import _safe_appdata_cache_root


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Optional cache root. Default: ./appdata/cache/datasets/",
    )
    args = ap.parse_args()

    root = args.cache_root.resolve() if args.cache_root else _safe_appdata_cache_root()
    payload: dict[str, Any] = {"cache_root": str(root), "datasets": []}
    if not root.exists():
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    for d in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        mp = d / "manifest.json"
        if not mp.exists():
            continue
        m = _load_manifest(mp)
        payload["datasets"].append(
            {
                "cache_key": str(m.get("cache_key") or d.name),
                "dataset_version_id": str(m.get("dataset_version_id") or ""),
                "row_count": int(m.get("row_count") or 0),
                "column_count_cached": int(m.get("column_count_cached") or 0),
                "format": str(m.get("format") or ""),
                "path": str(d),
            }
        )

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

