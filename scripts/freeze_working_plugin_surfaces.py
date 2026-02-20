#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (str(ROOT), str(SRC)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from statistic_harness.core.frozen_surfaces import (
    build_surface_record,
    default_contract_path,
    save_contract,
)
from statistic_harness.core.plugin_manager import PluginManager
from statistic_harness.core.storage import Storage
from statistic_harness.core.utils import now_iso


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Freeze known-good plugin surfaces from a completed run."
    )
    p.add_argument("--run-id", required=True, help="Run ID to source known-good plugin surfaces.")
    p.add_argument("--root", default=".", help="Repository root (default: .)")
    p.add_argument("--out", default="", help="Output contract path (default: docs/frozen_plugin_surfaces.contract.json)")
    p.add_argument(
        "--include-statuses",
        default="ok",
        help="Comma-separated plugin result statuses to lock (default: ok)",
    )
    p.add_argument(
        "--use-run-fingerprints",
        action="store_true",
        help="Use code_hash/settings_hash from the source run rows instead of current code/defaults.",
    )
    return p.parse_args()


def _status_set(raw: str) -> set[str]:
    return {part.strip().lower() for part in str(raw or "").split(",") if part.strip()}


def main() -> int:
    args = _args()
    root = Path(args.root).resolve()
    out_path = Path(args.out).resolve() if args.out else default_contract_path(root)
    include_statuses = _status_set(args.include_statuses) or {"ok"}

    storage = Storage(root / "appdata" / "state.sqlite", tenant_id=None, mode="ro", initialize=False)
    run_row = storage.fetch_run(args.run_id)
    if not run_row:
        raise SystemExit(f"Run not found: {args.run_id}")

    manager = PluginManager(root / "plugins")
    spec_map = {spec.plugin_id: spec for spec in manager.discover()}
    rows = storage.fetch_plugin_results(args.run_id)

    locked: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    for row in rows:
        plugin_id = str(row.get("plugin_id") or "")
        status = str(row.get("status") or "").strip().lower()
        if not plugin_id or status not in include_statuses:
            continue
        spec = spec_map.get(plugin_id)
        if spec is None:
            skipped[plugin_id] = "plugin_not_discovered"
            continue
        record = build_surface_record(
            spec,
            manager,
            code_hash=(str(row.get("code_hash") or "") or None) if args.use_run_fingerprints else None,
            settings_hash=(str(row.get("settings_hash") or "") or None) if args.use_run_fingerprints else None,
        )
        record["locked_from_run_id"] = args.run_id
        record["locked_from_status"] = status
        locked[plugin_id] = record

    payload = {
        "schema": "frozen_surfaces_contract.v1",
        "generated_at": now_iso(),
        "source_run_id": args.run_id,
        "source_dataset_version_id": run_row.get("dataset_version_id"),
        "plugins": locked,
    }
    save_contract(out_path, payload)

    summary = {
        "ok": True,
        "run_id": args.run_id,
        "out": str(out_path),
        "locked_count": int(len(locked)),
        "skipped_count": int(len(skipped)),
        "include_statuses": sorted(include_statuses),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
