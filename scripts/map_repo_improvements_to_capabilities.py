#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs" / "repo_improvements_catalog_v3.normalized.json"
DEFAULT_OUT = ROOT / "docs" / "repo_improvements_capability_map_v1.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _classification(item: dict[str, Any]) -> tuple[str, str]:
    category = str(item.get("category") or "").strip().lower()
    if "llm / agentic" in category:
        return (
            "deferred",
            "Deferred for phase-1 local-only policy: avoid runtime agentic/online dependencies.",
        )
    existing = item.get("existing_proposed_plugins") or []
    if isinstance(existing, list) and existing:
        return (
            "existing_plugin_enhancement",
            "At least one proposed plugin id already exists and can be enhanced safely.",
        )
    return (
        "new_plugin_scaffold",
        "No proposed plugin id exists yet; requires scaffold before implementation.",
    )


def build_payload(normalized_payload: dict[str, Any]) -> dict[str, Any]:
    rows = normalized_payload.get("catalog")
    if not isinstance(rows, list):
        raise ValueError("normalized payload missing list at key 'catalog'")
    mapped: list[dict[str, Any]] = []
    counts = {
        "existing_plugin_enhancement": 0,
        "new_plugin_scaffold": 0,
        "deferred": 0,
    }
    for item in rows:
        if not isinstance(item, dict):
            continue
        cls, rationale = _classification(item)
        counts[cls] += 1
        mapped.append(
            {
                "canonical_item_id": str(item.get("canonical_item_id") or ""),
                "category": str(item.get("category") or ""),
                "classification": cls,
                "classification_rationale": rationale,
                "target_paths": sorted([str(x) for x in (item.get("normalized_touchpoints") or []) if isinstance(x, str)]),
                "dependency_ids": sorted([str(x) for x in (item.get("dependency_ids") or []) if isinstance(x, str)]),
                "source_proposed_plugins": sorted(
                    [str(x) for x in (item.get("source_proposed_plugins") or []) if isinstance(x, str)]
                ),
                "existing_proposed_plugins": sorted(
                    [str(x) for x in (item.get("existing_proposed_plugins") or []) if isinstance(x, str)]
                ),
                "missing_proposed_plugins": sorted(
                    [str(x) for x in (item.get("missing_proposed_plugins") or []) if isinstance(x, str)]
                ),
                "priority_score": float(item.get("priority_score", 0.0)),
            }
        )
    mapped.sort(key=lambda row: str(row.get("canonical_item_id") or ""))
    return {
        "generated_by": "scripts/map_repo_improvements_to_capabilities.py",
        "source_normalized_catalog": "docs/repo_improvements_catalog_v3.normalized.json",
        "total_items": len(mapped),
        "classification_counts": counts,
        "items": mapped,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    normalized = _read_json(args.input)
    payload = build_payload(normalized)
    out = args.out.resolve()

    if args.verify:
        if not out.exists():
            return 2
        return 0 if _read_json(out) == payload else 2

    out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out, payload)
    print(f"out={out}")
    print(f"items={payload.get('total_items')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
