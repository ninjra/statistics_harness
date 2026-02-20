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
    contract_plugin_map,
    default_contract_path,
    evaluate_locked_surface,
    load_contract,
)
from statistic_harness.core.plugin_manager import PluginManager


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify locked plugin surfaces have not drifted."
    )
    p.add_argument("--root", default=".", help="Repository root (default: .)")
    p.add_argument("--contract", default="", help="Contract path (default: docs/frozen_plugin_surfaces.contract.json)")
    p.add_argument("--out", default="", help="Optional JSON report output path")
    return p.parse_args()


def main() -> int:
    args = _args()
    root = Path(args.root).resolve()
    contract_path = Path(args.contract).resolve() if args.contract else default_contract_path(root)
    payload = load_contract(contract_path)
    locked = contract_plugin_map(payload)

    manager = PluginManager(root / "plugins")
    spec_map = {spec.plugin_id: spec for spec in manager.discover()}

    missing_plugins: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    ok_count = 0

    for plugin_id in sorted(locked.keys()):
        expected = locked.get(plugin_id) or {}
        expected_surface_hash = str(expected.get("surface_hash") or "")
        if not expected_surface_hash:
            mismatches.append(
                {
                    "plugin_id": plugin_id,
                    "reason": "missing_expected_surface_hash",
                }
            )
            continue
        spec = spec_map.get(plugin_id)
        if spec is None:
            missing_plugins.append(
                {
                    "plugin_id": plugin_id,
                    "reason": "plugin_not_discovered",
                }
            )
            continue
        actual = build_surface_record(spec, manager)
        verdict = evaluate_locked_surface(
            plugin_id=plugin_id,
            expected_surface_hash=expected_surface_hash,
            actual_surface_hash=str(actual.get("surface_hash") or ""),
        )
        if verdict.get("ok"):
            ok_count += 1
            continue
        mismatches.append(
            {
                **verdict,
                "expected": expected,
                "actual": actual,
                "reason": "surface_hash_mismatch",
            }
        )

    report = {
        "ok": not missing_plugins and not mismatches,
        "contract_path": str(contract_path),
        "locked_count": int(len(locked)),
        "verified_ok_count": int(ok_count),
        "missing_plugins": missing_plugins,
        "mismatches": mismatches,
    }

    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
