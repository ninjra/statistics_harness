#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def _load(name: str) -> dict[str, Any]:
    path = DOCS / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _matrix_hard_gaps() -> dict[str, Any]:
    impl = _load("implementation_matrix.json")
    bind = _load("binding_implementation_matrix.json")
    redteam = _load("redteam_ids_matrix.json")
    plugins = _load("plugins_functionality_matrix.json")
    access = _load("plugin_data_access_matrix.json")
    sql = _load("sql_assist_adoption_matrix.json")
    instr = _load("full_instruction_coverage_report.json")

    return {
        "docs_missing_any_normative": bool(impl.get("missing_any_normative")),
        "binding_missing_any": bool(bind.get("missing_any")),
        "redteam_missing_required_ids": redteam.get("missing_required_ids") or [],
        "plugin_missing_dep_edges": plugins.get("missing_dep_edges") or {},
        "plugin_data_access_unclassified": [
            p.get("plugin_id")
            for p in (access.get("plugins") or [])
            if isinstance(p, dict) and bool(p.get("unclassified"))
        ],
        "instruction_coverage_items": instr.get("items") or [],
    }


def _soft_gaps() -> dict[str, Any]:
    sql = _load("sql_assist_adoption_matrix.json")
    recommended_not_using: list[str] = []
    optional_not_using: list[str] = []
    required_not_using: list[str] = []
    for row in sql.get("plugins") or []:
        if not isinstance(row, dict):
            continue
        uses_sql = bool(
            row.get("uses_sql_effective")
            or row.get("uses_sql")
            or row.get("uses_sql_exec")
        )
        intent = str(row.get("sql_intent") or "").strip()
        pid = str(row.get("plugin_id") or "").strip()
        if not pid or uses_sql:
            continue
        if intent == "required":
            required_not_using.append(pid)
        elif intent == "recommended":
            recommended_not_using.append(pid)
        elif intent == "optional":
            optional_not_using.append(pid)
    return {
        "required_sql_not_using": sorted(required_not_using),
        "recommended_sql_not_using": sorted(recommended_not_using),
        "optional_sql_not_using": sorted(optional_not_using),
    }


def _payload() -> dict[str, Any]:
    hard = _matrix_hard_gaps()
    soft = _soft_gaps()
    has_hard_gaps = any(
        [
            bool(hard.get("docs_missing_any_normative")),
            bool(hard.get("binding_missing_any")),
            bool(hard.get("redteam_missing_required_ids")),
            bool(hard.get("plugin_missing_dep_edges")),
            bool(hard.get("plugin_data_access_unclassified")),
            bool(hard.get("instruction_coverage_items")),
        ]
    )
    return {
        "generated_by": "scripts/full_repo_misses.py",
        "has_hard_gaps": has_hard_gaps,
        "hard_gaps": hard,
        "soft_gaps": soft,
        "summary": {
            "hard_gap_count": int(
                len(hard.get("redteam_missing_required_ids") or [])
                + len(hard.get("plugin_data_access_unclassified") or [])
                + len(hard.get("instruction_coverage_items") or [])
                + (1 if hard.get("docs_missing_any_normative") else 0)
                + (1 if hard.get("binding_missing_any") else 0)
                + (1 if hard.get("plugin_missing_dep_edges") else 0)
            ),
            "required_sql_not_using_count": len(soft.get("required_sql_not_using") or []),
            "recommended_sql_not_using_count": len(soft.get("recommended_sql_not_using") or []),
            "optional_sql_not_using_count": len(soft.get("optional_sql_not_using") or []),
        },
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Full Repo Misses")
    lines.append("")
    lines.append(f"- has_hard_gaps: {payload.get('has_hard_gaps')}")
    lines.append("")
    lines.append("## Hard Gaps")
    hard = payload.get("hard_gaps") or {}
    lines.append(f"- docs_missing_any_normative: {hard.get('docs_missing_any_normative')}")
    lines.append(f"- binding_missing_any: {hard.get('binding_missing_any')}")
    lines.append(f"- redteam_missing_required_ids: {hard.get('redteam_missing_required_ids')}")
    lines.append(f"- plugin_missing_dep_edges: {hard.get('plugin_missing_dep_edges')}")
    lines.append(f"- plugin_data_access_unclassified: {hard.get('plugin_data_access_unclassified')}")
    lines.append("")
    if hard.get("instruction_coverage_items"):
        lines.append("### Instruction Coverage Items")
        for item in hard.get("instruction_coverage_items") or []:
            lines.append(
                f"- `{item.get('source')}:{item.get('line_no')}` {item.get('item_type')}: {item.get('text')}"
            )
        lines.append("")
    lines.append("## Soft Gaps")
    soft = payload.get("soft_gaps") or {}
    lines.append(f"- required_sql_not_using: {soft.get('required_sql_not_using')}")
    lines.append(f"- recommended_sql_not_using_count: {len(soft.get('recommended_sql_not_using') or [])}")
    lines.append(f"- optional_sql_not_using_count: {len(soft.get('optional_sql_not_using') or [])}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="docs/full_repo_misses.json")
    ap.add_argument("--out-md", default="docs/full_repo_misses.md")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    payload = _payload()
    out_json = (ROOT / args.out_json).resolve()
    out_md = (ROOT / args.out_md).resolve()
    json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    md_text = _markdown(payload)

    if args.verify:
        if not out_json.exists() or out_json.read_text(encoding="utf-8") != json_text:
            return 2
        if not out_md.exists() or out_md.read_text(encoding="utf-8") != md_text:
            return 2
        return 0

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json_text, encoding="utf-8")
    out_md.write_text(md_text, encoding="utf-8")
    print(f"out_json={out_json}")
    print(f"out_md={out_md}")
    print(f"has_hard_gaps={payload.get('has_hard_gaps')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
