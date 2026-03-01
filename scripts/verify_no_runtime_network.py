#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from statistic_harness.core.plugin_manager import PluginManager


ROOT = Path(__file__).resolve().parents[1]


def _load_manifest_plugins(run_id: str, runs_root: Path) -> set[str]:
    path = runs_root / run_id / "run_manifest.json"
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    plugins = payload.get("plugins") if isinstance(payload, dict) else None
    if not isinstance(plugins, list):
        return set()
    out: set[str] = set()
    for row in plugins:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("plugin_id") or "").strip()
        if pid:
            out.add(pid)
    return out


def verify_no_runtime_network(
    *,
    plugins_dir: Path,
    run_id: str | None,
    runs_root: Path,
) -> dict[str, Any]:
    manager = PluginManager(plugins_dir)
    specs = manager.discover()
    run_plugin_ids = _load_manifest_plugins(run_id, runs_root) if run_id else set()
    violations: list[dict[str, str]] = []

    for spec in specs:
        if run_plugin_ids and spec.plugin_id not in run_plugin_ids:
            continue
        plugin_type = str(spec.type or "").strip().lower()
        if plugin_type not in {"analysis", "transform", "profile", "report", "llm", "planner"}:
            continue
        no_network = bool((spec.sandbox or {}).get("no_network"))
        if not no_network:
            violations.append(
                {
                    "plugin_id": spec.plugin_id,
                    "plugin_type": plugin_type,
                    "reason": "SANDBOX_NO_NETWORK_NOT_TRUE",
                }
            )

    return {
        "schema_version": "no_runtime_network_contract.v1",
        "run_id": run_id or None,
        "ok": len(violations) == 0,
        "violation_count": len(violations),
        "violations": violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify plugin no-network contract.")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--plugins-dir", default=str(ROOT / "plugins"))
    parser.add_argument("--runs-root", default=str(ROOT / "appdata" / "runs"))
    parser.add_argument("--out-json", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    payload = verify_no_runtime_network(
        plugins_dir=Path(str(args.plugins_dir)),
        run_id=str(args.run_id).strip() or None,
        runs_root=Path(str(args.runs_root)),
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out_json = str(args.out_json).strip()
    if out_json:
        out_path = Path(out_json)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if bool(args.strict) and not bool(payload.get("ok")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

