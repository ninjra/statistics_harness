#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NORMALIZED_INPUT = ROOT / "docs" / "repo_improvements_catalog_v3.normalized.json"
DEFAULT_MAP_INPUT = ROOT / "docs" / "repo_improvements_capability_map_v1.json"
DEFAULT_OUT_JSON = ROOT / "docs" / "repo_improvements_execution_plan_v1.json"
DEFAULT_OUT_MD = ROOT / "docs" / "repo_improvements_execution_plan_v1.md"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _depth(node: str, deps: dict[str, list[str]], memo: dict[str, int], visiting: set[str]) -> int:
    if node in memo:
        return memo[node]
    if node in visiting:
        return 0
    visiting.add(node)
    parents = deps.get(node, [])
    if not parents:
        memo[node] = 0
    else:
        memo[node] = 1 + max(_depth(dep, deps, memo, visiting) for dep in parents)
    visiting.remove(node)
    return memo[node]


def build_payload(normalized_payload: dict[str, Any], mapped_payload: dict[str, Any]) -> dict[str, Any]:
    normalized_items = normalized_payload.get("catalog")
    mapped_items = mapped_payload.get("items")
    if not isinstance(normalized_items, list):
        raise ValueError("normalized payload missing list at key 'catalog'")
    if not isinstance(mapped_items, list):
        raise ValueError("capability map payload missing list at key 'items'")

    map_by_id = {
        str(item.get("canonical_item_id") or ""): item
        for item in mapped_items
        if isinstance(item, dict)
    }
    nodes = sorted([str(item.get("canonical_item_id") or "") for item in normalized_items if isinstance(item, dict)])
    node_set = set(nodes)
    deps: dict[str, list[str]] = {}
    unknown_deps: dict[str, list[str]] = {}
    for item in normalized_items:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("canonical_item_id") or "")
        item_deps = sorted([str(x) for x in (item.get("dependency_ids") or []) if isinstance(x, str)])
        known = [x for x in item_deps if x in node_set]
        unknown = [x for x in item_deps if x not in node_set]
        deps[cid] = known
        if unknown:
            unknown_deps[cid] = unknown

    depth_memo: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for item in normalized_items:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("canonical_item_id") or "")
        mapped = map_by_id.get(cid, {})
        cls = str(mapped.get("classification") or "new_plugin_scaffold")
        base = float(item.get("priority_score", 0.0))
        if cls == "deferred":
            base -= 1.0
        elif cls == "existing_plugin_enhancement":
            base += 0.15
        depth = _depth(cid, deps, depth_memo, set())
        rows.append(
            {
                "canonical_item_id": cid,
                "category": str(item.get("category") or ""),
                "classification": cls,
                "classification_rationale": str(mapped.get("classification_rationale") or ""),
                "priority_score": round(base, 4),
                "dependency_depth": int(depth),
                "dependency_ids": deps.get(cid, []),
                "unknown_dependency_ids": unknown_deps.get(cid, []),
                "target_paths": sorted([str(x) for x in (mapped.get("target_paths") or []) if isinstance(x, str)]),
                "source_proposed_plugins": sorted(
                    [str(x) for x in (item.get("source_proposed_plugins") or []) if isinstance(x, str)]
                ),
                "missing_proposed_plugins": sorted(
                    [str(x) for x in (item.get("missing_proposed_plugins") or []) if isinstance(x, str)]
                ),
            }
        )
    rows.sort(key=lambda r: (-float(r["priority_score"]), int(r["dependency_depth"]), str(r["canonical_item_id"])))

    for idx, row in enumerate(rows):
        if idx < 4:
            row["wave"] = "wave_1"
        elif idx < 7:
            row["wave"] = "wave_2"
        else:
            row["wave"] = "wave_3"
        row["wave_rank"] = idx + 1

    wave_1_scaffold_ids = sorted(
        {
            pid
            for row in rows
            if row.get("wave") == "wave_1"
            for pid in (row.get("missing_proposed_plugins") or [])
            if isinstance(pid, str) and pid
        }
    )
    return {
        "generated_by": "scripts/plan_repo_improvements_rollout.py",
        "source_normalized_catalog": "docs/repo_improvements_catalog_v3.normalized.json",
        "source_capability_map": "docs/repo_improvements_capability_map_v1.json",
        "total_items": len(rows),
        "wave_counts": {
            "wave_1": sum(1 for row in rows if row.get("wave") == "wave_1"),
            "wave_2": sum(1 for row in rows if row.get("wave") == "wave_2"),
            "wave_3": sum(1 for row in rows if row.get("wave") == "wave_3"),
        },
        "wave_1_scaffold_plugin_ids": wave_1_scaffold_ids,
        "items": rows,
    }


def build_markdown(payload: dict[str, Any]) -> str:
    items = payload.get("items") or []
    lines: list[str] = []
    lines.append("# Repo Improvements Execution Plan v1")
    lines.append("")
    lines.append("Generated by `scripts/plan_repo_improvements_rollout.py`.")
    lines.append("")
    lines.append(f"- Total items: {payload.get('total_items')}")
    counts = payload.get("wave_counts") or {}
    lines.append(f"- wave_1: {counts.get('wave_1', 0)}")
    lines.append(f"- wave_2: {counts.get('wave_2', 0)}")
    lines.append(f"- wave_3: {counts.get('wave_3', 0)}")
    lines.append("")
    lines.append("| Rank | Canonical Item | Wave | Classification | Score | Depth |")
    lines.append("|---:|---|---|---|---:|---:|")
    for row in items:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("wave_rank")),
                    f"`{row.get('canonical_item_id')}`",
                    str(row.get("wave")),
                    str(row.get("classification")),
                    f"{float(row.get('priority_score', 0.0)):.4f}",
                    str(row.get("dependency_depth")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normalized-input", type=Path, default=DEFAULT_NORMALIZED_INPUT)
    ap.add_argument("--map-input", type=Path, default=DEFAULT_MAP_INPUT)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    normalized = _read_json(args.normalized_input)
    mapped = _read_json(args.map_input)
    payload = build_payload(normalized, mapped)
    md_text = build_markdown(payload)

    out_json = args.out_json.resolve()
    out_md = args.out_md.resolve()

    if args.verify:
        if not out_json.exists() or not out_md.exists():
            return 2
        if _read_json(out_json) != payload:
            return 2
        return 0 if out_md.read_text(encoding="utf-8") == md_text else 2

    out_json.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out_json, payload)
    out_md.write_text(md_text, encoding="utf-8")
    print(f"out_json={out_json}")
    print(f"out_md={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
