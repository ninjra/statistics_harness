#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.plugin_data_access_matrix import (  # type: ignore[no-redef]
        CONTRACT_ARTIFACT_ONLY,
        CONTRACT_ORCHESTRATION_ONLY,
        CONTRACT_DATASET_LOADER,
        CONTRACT_ITER_BATCHES,
        CONTRACT_SQL_DIRECT,
        generate as generate_access_matrix,
    )
else:
    from scripts.plugin_data_access_matrix import (
        CONTRACT_ARTIFACT_ONLY,
        CONTRACT_ORCHESTRATION_ONLY,
        CONTRACT_DATASET_LOADER,
        CONTRACT_ITER_BATCHES,
        CONTRACT_SQL_DIRECT,
        generate as generate_access_matrix,
    )

ROOT = Path(__file__).resolve().parents[1]
INTENT_OVERRIDE_PATH = ROOT / "docs" / "sql_assist_intent_overrides.json"
ALLOWED_INTENTS = {"required", "recommended", "optional", "not_applicable"}


@dataclass(frozen=True)
class Row:
    plugin_id: str
    plugin_type: str
    uses_sql: bool
    uses_sql_exec: bool
    uses_sql_via_loader: bool
    uses_sql_direct: bool
    uses_sql_effective: bool
    sql_intent: str
    sql_intent_source: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(_read_text(manifest_path))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _plugin_type(manifest_payload: dict[str, Any]) -> str:
    value = manifest_payload.get("type")
    return value.strip() if isinstance(value, str) else ""


def _load_intent_overrides(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    rows = payload.get("intents") if isinstance(payload, dict) else None
    if not isinstance(rows, dict):
        return {}
    out: dict[str, str] = {}
    for plugin_id, intent in rows.items():
        if not isinstance(plugin_id, str) or not isinstance(intent, str):
            continue
        value = intent.strip()
        if value in ALLOWED_INTENTS:
            out[plugin_id] = value
    return out


def _default_intent(plugin_type: str, access_contracts: set[str]) -> str:
    if access_contracts and access_contracts.issubset({CONTRACT_ARTIFACT_ONLY, CONTRACT_ORCHESTRATION_ONLY}):
        return "not_applicable"
    ptype = plugin_type.strip().lower()
    if ptype == "analysis":
        return "recommended"
    if ptype == "transform":
        return "optional"
    if ptype in {"report", "planner", "llm", "ingest", "profile"}:
        return "not_applicable"
    return "optional"


def generate(plugins_root: Path) -> list[Row]:
    overrides = _load_intent_overrides(INTENT_OVERRIDE_PATH)
    access_rows = {item.plugin_id: item for item in generate_access_matrix(plugins_root)}
    out: list[Row] = []
    for pdir in sorted([p for p in plugins_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        manifest = pdir / "plugin.yaml"
        entry = pdir / "plugin.py"
        if not manifest.exists() or not entry.exists():
            continue
        manifest_payload = _manifest(manifest)
        ptype = _plugin_type(manifest_payload)
        text = _read_text(entry)
        if pdir.name in overrides:
            intent = overrides[pdir.name]
            intent_source = "override"
        else:
            access_row = access_rows.get(pdir.name)
            contracts = set(access_row.access_contracts) if access_row else set()
            intent = _default_intent(ptype, contracts)
            intent_source = "default"
        access_row = access_rows.get(pdir.name)
        access_contracts = set(access_row.access_contracts) if access_row else set()
        uses_sql_via_loader = bool(
            CONTRACT_DATASET_LOADER in access_contracts
            or CONTRACT_ITER_BATCHES in access_contracts
        )
        uses_sql_direct = bool(
            CONTRACT_SQL_DIRECT in access_contracts
        )
        uses_sql = ("ctx.sql" in text)
        uses_sql_exec = ("ctx.sql_exec" in text)
        uses_sql_effective = bool(uses_sql or uses_sql_exec or uses_sql_via_loader or uses_sql_direct)

        out.append(
            Row(
                plugin_id=pdir.name,
                plugin_type=ptype,
                uses_sql=uses_sql,
                uses_sql_exec=uses_sql_exec,
                uses_sql_via_loader=uses_sql_via_loader,
                uses_sql_direct=uses_sql_direct,
                uses_sql_effective=uses_sql_effective,
                sql_intent=intent,
                sql_intent_source=intent_source,
            )
        )
    return out


def _as_json(items: list[Row]) -> dict[str, Any]:
    by_intent: dict[str, dict[str, int]] = {}
    for intent in sorted(ALLOWED_INTENTS):
        by_intent[intent] = {"count": 0, "using_sql": 0}
    for item in items:
        bucket = by_intent[item.sql_intent]
        bucket["count"] += 1
        if item.uses_sql_effective:
            bucket["using_sql"] += 1
    coverage = {}
    for intent, bucket in by_intent.items():
        count = bucket["count"]
        using = bucket["using_sql"]
        coverage[intent] = (float(using) / float(count)) if count else 0.0
    return {
        "plugin_count": len(items),
        "uses_sql": sum(1 for i in items if i.uses_sql),
        "uses_sql_exec": sum(1 for i in items if i.uses_sql_exec),
        "uses_sql_via_loader": sum(1 for i in items if i.uses_sql_via_loader),
        "uses_sql_direct": sum(1 for i in items if i.uses_sql_direct),
        "uses_sql_effective": sum(1 for i in items if i.uses_sql_effective),
        "intent_counts": {k: v["count"] for k, v in by_intent.items()},
        "intent_using_sql_counts": {k: v["using_sql"] for k, v in by_intent.items()},
        "intent_sql_coverage": coverage,
        "plugins": [
            {
                "plugin_id": i.plugin_id,
                "plugin_type": i.plugin_type,
                "uses_sql": bool(i.uses_sql),
                "uses_sql_exec": bool(i.uses_sql_exec),
                "uses_sql_via_loader": bool(i.uses_sql_via_loader),
                "uses_sql_direct": bool(i.uses_sql_direct),
                "uses_sql_effective": bool(i.uses_sql_effective),
                "sql_intent": i.sql_intent,
                "sql_intent_source": i.sql_intent_source,
            }
            for i in items
        ],
    }


def _as_md(items: list[Row]) -> str:
    lines: list[str] = []
    lines.append("# SQL Assist Adoption Matrix")
    lines.append("")
    lines.append("Generated by `scripts/sql_assist_adoption_matrix.py`.")
    lines.append("")
    lines.append("| Plugin | Type | intent | intent_source | ctx.sql | ctx.sql_exec | via_loader | direct_sql | sql_effective |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    for i in items:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{i.plugin_id}`",
                    i.plugin_type or "",
                    i.sql_intent,
                    i.sql_intent_source,
                    str(int(i.uses_sql)),
                    str(int(i.uses_sql_exec)),
                    str(int(i.uses_sql_via_loader)),
                    str(int(i.uses_sql_direct)),
                    str(int(i.uses_sql_effective)),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugins-root", default="plugins")
    ap.add_argument("--out-json", default="docs/sql_assist_adoption_matrix.json")
    ap.add_argument("--out-md", default="docs/sql_assist_adoption_matrix.md")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    items = generate((ROOT / args.plugins_root).resolve())
    payload = _as_json(items)
    out_json = (ROOT / args.out_json).resolve()
    out_md = (ROOT / args.out_md).resolve()
    json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    md_text = _as_md(items)

    if args.verify:
        if not out_json.exists() or out_json.read_text(encoding="utf-8") != json_text:
            return 2
        if not out_md.exists() or out_md.read_text(encoding="utf-8") != md_text:
            return 2
        return 0

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json_text, encoding="utf-8")
    out_md.write_text(md_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
