#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT / "docs" / "repo_improvements_catalog_v3.json"
DEFAULT_CANONICAL_OUTPUT_PATH = ROOT / "docs" / "repo_improvements_catalog_v3.canonical.json"
DEFAULT_REDUCTION_OUTPUT_PATH = ROOT / "docs" / "repo_improvements_catalog_v3.reduction_report.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalized_list(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list):
        return tuple()
    return tuple(str(v).strip() for v in values)


def _impact_signature(item: dict[str, Any]) -> tuple[tuple[str, float], ...]:
    impact = item.get("four_pillars_expected_impact")
    if not isinstance(impact, dict):
        return tuple()
    out: list[tuple[str, float]] = []
    for key in sorted(impact):
        value = impact.get(key)
        if isinstance(value, (int, float)):
            out.append((str(key), float(value)))
    return tuple(out)


def _group_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(item.get("category") or "").strip(),
        str(item.get("description") or "").strip(),
        _normalized_list(item.get("implementation_steps")),
        _normalized_list(item.get("acceptance_criteria")),
        _normalized_list(item.get("repo_touchpoints")),
        _impact_signature(item),
        bool(item.get("deterministic", False)),
        bool(item.get("sandbox_compliant", False)),
    )


def _title_template(titles: list[str]) -> str:
    cleaned = [str(t).strip() for t in titles if str(t).strip()]
    if not cleaned:
        return ""
    base = [re.sub(r"\s+#\d+\s*$", "", t) for t in cleaned]
    if len(set(base)) == 1:
        return f"{base[0]} #<n>"
    return cleaned[0]


def _cluster_id(category: str, source_ids: list[str]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", category.strip().lower()).strip("-")
    digest_input = "|".join(source_ids).encode("utf-8")
    suffix = hashlib.sha256(digest_input).hexdigest()[:8]
    return f"{slug}-{suffix}"


def _mean_impact(items: list[dict[str, Any]]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for item in items:
        impact = item.get("four_pillars_expected_impact")
        if not isinstance(impact, dict):
            continue
        for key, value in impact.items():
            if not isinstance(value, (int, float)):
                continue
            k = str(key)
            sums[k] = sums.get(k, 0.0) + float(value)
            counts[k] = counts.get(k, 0) + 1
    out: dict[str, float] = {}
    for key in sorted(sums):
        out[key] = round(sums[key] / float(counts[key]), 4)
    return out


def build_outputs_from_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_items = payload.get("catalog")
    if not isinstance(raw_items, list):
        raise ValueError("catalog payload missing list at key 'catalog'")

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        groups[_group_key(item)].append(item)

    def _first_source_id(group_items: list[dict[str, Any]]) -> str:
        ids = [str(it.get("id") or "") for it in group_items]
        return sorted(ids)[0] if ids else ""

    canonical_entries: list[dict[str, Any]] = []
    source_to_canonical: dict[str, str] = {}
    sorted_groups = sorted(groups.values(), key=_first_source_id)
    for index, group_items in enumerate(sorted_groups, start=1):
        source_ids = sorted(str(it.get("id") or "") for it in group_items if str(it.get("id") or ""))
        source_plugins = sorted(
            {
                str(plugin_id).strip()
                for it in group_items
                for plugin_id in (it.get("proposed_plugins") or [])
                if str(plugin_id).strip()
            }
        )
        first = group_items[0]
        category = str(first.get("category") or "").strip()
        canonical_id = f"CANONICAL_{index:03d}"
        for source_id in source_ids:
            source_to_canonical[source_id] = canonical_id

        entry = {
            "canonical_item_id": canonical_id,
            "cluster_id": _cluster_id(category, source_ids),
            "category": category,
            "title_template": _title_template([str(it.get("title") or "") for it in group_items]),
            "description": str(first.get("description") or "").strip(),
            "source_item_count": len(source_ids),
            "source_item_ids": source_ids,
            "source_proposed_plugins": source_plugins,
            "repo_touchpoints": list(_normalized_list(first.get("repo_touchpoints"))),
            "implementation_steps": list(_normalized_list(first.get("implementation_steps"))),
            "acceptance_criteria": list(_normalized_list(first.get("acceptance_criteria"))),
            "dependencies": sorted(
                {
                    str(dep).strip()
                    for it in group_items
                    for dep in (it.get("dependencies") or [])
                    if str(dep).strip()
                }
            ),
            "four_pillars_expected_impact": _mean_impact(group_items),
            "deterministic": bool(all(bool(it.get("deterministic", False)) for it in group_items)),
            "sandbox_compliant": bool(all(bool(it.get("sandbox_compliant", False)) for it in group_items)),
        }
        canonical_entries.append(entry)

    source_catalog = "docs/repo_improvements_catalog_v3.json"
    generated_utc = str(payload.get("generated_utc") or "")
    total_source = len([item for item in raw_items if isinstance(item, dict)])
    total_canonical = len(canonical_entries)
    reduction_percent = 0.0
    if total_source > 0:
        reduction_percent = round(100.0 * (1.0 - (float(total_canonical) / float(total_source))), 4)

    canonical_payload: dict[str, Any] = {
        "generated_utc": generated_utc,
        "source_catalog": source_catalog,
        "total_source_items": total_source,
        "total_canonical_items": total_canonical,
        "catalog": canonical_entries,
    }

    reduction_payload: dict[str, Any] = {
        "generated_utc": generated_utc,
        "source_catalog": source_catalog,
        "total_source_items": total_source,
        "total_canonical_items": total_canonical,
        "reduction_percent": reduction_percent,
        "source_to_canonical": {key: source_to_canonical[key] for key in sorted(source_to_canonical)},
        "clusters": [
            {
                "canonical_item_id": entry["canonical_item_id"],
                "cluster_id": entry["cluster_id"],
                "category": entry["category"],
                "source_item_count": entry["source_item_count"],
                "source_item_ids": entry["source_item_ids"],
            }
            for entry in canonical_entries
        ],
    }
    return canonical_payload, reduction_payload


def _verify_or_write(path: Path, payload: dict[str, Any], verify: bool) -> int:
    if verify:
        if not path.exists():
            print(f"missing: {path}")
            return 1
        existing = _read_json(path)
        if existing != payload:
            print(f"stale: {path}")
            return 1
        return 0
    _write_json(path, payload)
    print(f"wrote: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Canonicalize repo improvements catalog")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--canonical-output", type=Path, default=DEFAULT_CANONICAL_OUTPUT_PATH)
    parser.add_argument("--reduction-output", type=Path, default=DEFAULT_REDUCTION_OUTPUT_PATH)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)

    payload = _read_json(args.input)
    canonical_payload, reduction_payload = build_outputs_from_payload(payload)

    rc1 = _verify_or_write(args.canonical_output, canonical_payload, args.verify)
    rc2 = _verify_or_write(args.reduction_output, reduction_payload, args.verify)
    return 0 if (rc1 == 0 and rc2 == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
