#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jsonschema import validate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_INPUT = ROOT / "docs" / "repo_improvements_catalog_v3.json"
DEFAULT_CANONICAL_INPUT = ROOT / "docs" / "repo_improvements_catalog_v3.canonical.json"
DEFAULT_TOUCHPOINT_MAP = ROOT / "docs" / "repo_improvements_touchpoint_map.json"
DEFAULT_RAW_SCHEMA = ROOT / "docs" / "repo_improvements_catalog.raw.schema.json"
DEFAULT_NORMALIZED_SCHEMA = ROOT / "docs" / "repo_improvements_catalog.normalized.schema.json"
DEFAULT_OUT = ROOT / "docs" / "repo_improvements_catalog_v3.normalized.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _plugin_ids_on_disk() -> set[str]:
    plugins_root = ROOT / "plugins"
    return {p.name for p in plugins_root.iterdir() if p.is_dir()}


def _resolve_touchpoints(repo_touchpoints: list[str], touchpoint_map: dict[str, Any]) -> list[str]:
    map_rows = touchpoint_map.get("touchpoint_map")
    if not isinstance(map_rows, dict):
        raise ValueError("touchpoint_map missing key 'touchpoint_map'")
    resolved: set[str] = set()
    for raw_tp in repo_touchpoints:
        mapped = map_rows.get(raw_tp)
        if mapped is None:
            raise ValueError(f"unmapped touchpoint: {raw_tp}")
        mapped_list = [mapped] if isinstance(mapped, str) else mapped
        if not isinstance(mapped_list, list):
            raise ValueError(f"invalid mapping for touchpoint: {raw_tp}")
        for value in mapped_list:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"invalid mapped touchpoint value for {raw_tp}")
            rel = value.strip().replace("\\", "/")
            if not (ROOT / rel).exists():
                raise ValueError(f"mapped touchpoint does not exist: {rel}")
            resolved.add(rel)
    return sorted(resolved)


def _impact_inputs(item: dict[str, Any]) -> dict[str, Any]:
    impact = item.get("four_pillars_expected_impact")
    if not isinstance(impact, dict):
        impact = {}
    return {
        "performant": float(impact.get("performant", 0.0)),
        "accurate": float(impact.get("accurate", 0.0)),
        "secure": float(impact.get("secure", 0.0)),
        "citable": float(impact.get("citable", 0.0)),
    }


def build_payload(
    *,
    raw_payload: dict[str, Any],
    canonical_payload: dict[str, Any],
    touchpoint_map: dict[str, Any],
) -> dict[str, Any]:
    plugin_ids = _plugin_ids_on_disk()
    catalog = canonical_payload.get("catalog")
    if not isinstance(catalog, list):
        raise ValueError("canonical payload missing list at key 'catalog'")

    normalized_items: list[dict[str, Any]] = []
    for item in catalog:
        if not isinstance(item, dict):
            continue
        touchpoints = [str(x) for x in (item.get("repo_touchpoints") or []) if isinstance(x, str)]
        normalized_touchpoints = _resolve_touchpoints(touchpoints, touchpoint_map)
        proposed = [str(x) for x in (item.get("source_proposed_plugins") or []) if isinstance(x, str) and str(x).strip()]
        proposed = sorted({x.strip() for x in proposed})
        existing = sorted([pid for pid in proposed if pid in plugin_ids])
        missing = sorted([pid for pid in proposed if pid not in plugin_ids])
        deps = sorted({str(x) for x in (item.get("dependencies") or []) if isinstance(x, str) and str(x).strip()})
        impacts = _impact_inputs(item)
        dep_count = len(deps)
        score = (
            impacts["performant"]
            + impacts["accurate"]
            + impacts["secure"]
            + impacts["citable"]
        ) / 4.0
        score = round(score - (0.05 * dep_count), 4)
        normalized_items.append(
            {
                "canonical_item_id": str(item.get("canonical_item_id") or ""),
                "cluster_id": str(item.get("cluster_id") or ""),
                "category": str(item.get("category") or ""),
                "source_item_ids": sorted([str(x) for x in (item.get("source_item_ids") or []) if isinstance(x, str)]),
                "source_proposed_plugins": proposed,
                "existing_proposed_plugins": existing,
                "missing_proposed_plugins": missing,
                "normalized_touchpoints": normalized_touchpoints,
                "dependency_ids": deps,
                "implementation_status": "todo",
                "priority_score_inputs": {
                    **impacts,
                    "dependency_count": dep_count,
                },
                "priority_score": score,
                "deterministic": bool(item.get("deterministic", False)),
                "sandbox_compliant": bool(item.get("sandbox_compliant", False)),
            }
        )
    normalized_items.sort(key=lambda row: str(row.get("canonical_item_id") or ""))
    return {
        "generated_utc": str(canonical_payload.get("generated_utc") or raw_payload.get("generated_utc") or ""),
        "source_catalog": "docs/repo_improvements_catalog_v3.json",
        "source_canonical_catalog": "docs/repo_improvements_catalog_v3.canonical.json",
        "total_items": len(normalized_items),
        "catalog": normalized_items,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-input", type=Path, default=DEFAULT_RAW_INPUT)
    ap.add_argument("--canonical-input", type=Path, default=DEFAULT_CANONICAL_INPUT)
    ap.add_argument("--touchpoint-map", type=Path, default=DEFAULT_TOUCHPOINT_MAP)
    ap.add_argument("--raw-schema", type=Path, default=DEFAULT_RAW_SCHEMA)
    ap.add_argument("--normalized-schema", type=Path, default=DEFAULT_NORMALIZED_SCHEMA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    raw_payload = _read_json(args.raw_input)
    canonical_payload = _read_json(args.canonical_input)
    touchpoint_map = _read_json(args.touchpoint_map)
    raw_schema = _read_json(args.raw_schema)
    normalized_schema = _read_json(args.normalized_schema)

    validate(instance=raw_payload, schema=raw_schema)
    payload = build_payload(
        raw_payload=raw_payload,
        canonical_payload=canonical_payload,
        touchpoint_map=touchpoint_map,
    )
    validate(instance=payload, schema=normalized_schema)

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
