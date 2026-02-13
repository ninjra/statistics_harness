#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs" / "repo_improvements_execution_plan_v1.json"
DEFAULT_OUT = ROOT / "docs" / "repo_improvements_dependency_validation_v1.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _find_cycles(graph: dict[str, list[str]]) -> list[list[str]]:
    cycles: list[list[str]] = []
    seen: set[str] = set()
    active: list[str] = []
    active_set: set[str] = set()

    def dfs(node: str) -> None:
        if node in active_set:
            idx = active.index(node)
            cycle = active[idx:] + [node]
            if cycle not in cycles:
                cycles.append(cycle)
            return
        if node in seen:
            return
        seen.add(node)
        active.append(node)
        active_set.add(node)
        for nxt in graph.get(node, []):
            dfs(nxt)
        active.pop()
        active_set.remove(node)

    for node in sorted(graph):
        dfs(node)
    return cycles


def validate_plan_payload(plan_payload: dict[str, Any]) -> dict[str, Any]:
    items = plan_payload.get("items")
    if not isinstance(items, list):
        raise ValueError("plan payload missing list at key 'items'")

    ids = [str(item.get("canonical_item_id") or "") for item in items if isinstance(item, dict)]
    id_set = set(ids)
    unknown_deps: dict[str, list[str]] = {}
    graph: dict[str, list[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("canonical_item_id") or "")
        deps = sorted([str(x) for x in (item.get("dependency_ids") or []) if isinstance(x, str) and str(x).strip()])
        graph[cid] = [dep for dep in deps if dep in id_set]
        missing = [dep for dep in deps if dep not in id_set]
        if missing:
            unknown_deps[cid] = missing

    cycles = _find_cycles(graph)
    has_errors = bool(unknown_deps or cycles)
    return {
        "generated_by": "scripts/validate_repo_improvement_dependencies.py",
        "source_execution_plan": "docs/repo_improvements_execution_plan_v1.json",
        "total_items": len(ids),
        "unknown_dependencies": unknown_deps,
        "cycles": cycles,
        "has_errors": has_errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    payload = _read_json(args.input)
    validation = validate_plan_payload(payload)
    out = args.out.resolve()

    if args.verify:
        if validation.get("has_errors"):
            return 2
        if not out.exists():
            return 2
        return 0 if _read_json(out) == validation else 2

    out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out, validation)
    print(f"out={out}")
    print(f"has_errors={validation.get('has_errors')}")
    return 0 if not validation.get("has_errors") else 2


if __name__ == "__main__":
    raise SystemExit(main())
