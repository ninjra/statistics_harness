#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import yaml

from statistic_harness.core.actionability_explanations import derive_reason_code


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAXONOMY = ROOT / "docs" / "plugin_class_taxonomy.yaml"
DEFAULT_JSON = ROOT / "docs" / "plugin_class_actionability_matrix.json"
DEFAULT_MD = ROOT / "docs" / "plugin_class_actionability_matrix.md"


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_plugin_manifests() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest in sorted((ROOT / "plugins").glob("*/plugin.yaml")):
        try:
            payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        plugin_id = str(payload.get("id") or manifest.parent.name).strip()
        if not plugin_id:
            continue
        rows.append(
            {
                "plugin_id": plugin_id,
                "plugin_type": str(payload.get("type") or "").strip(),
                "name": str(payload.get("name") or plugin_id).strip(),
                "depends_on": [str(v).strip() for v in (payload.get("depends_on") or []) if str(v).strip()],
            }
        )
    return rows


def _latest_completed_run_id() -> str:
    db = ROOT / "appdata" / "state.sqlite"
    if not db.exists():
        return ""
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            "SELECT run_id FROM runs WHERE status = 'completed' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    return str(row[0]) if row and row[0] else ""


def _report_for_run(run_id: str) -> dict[str, Any]:
    if not run_id:
        return {}
    path = ROOT / "appdata" / "runs" / run_id / "report.json"
    if not path.exists():
        return {}
    return _read_json(path)


def _plugin_examples(report: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    recs = report.get("recommendations") if isinstance(report.get("recommendations"), dict) else {}
    items = recs.get("items") if isinstance(recs.get("items"), list) else []
    explanations_block = recs.get("explanations") if isinstance(recs.get("explanations"), dict) else {}
    explanations = explanations_block.get("items") if isinstance(explanations_block.get("items"), list) else []
    rec_by_plugin: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        plugin_id = str(item.get("plugin_id") or "").strip()
        if plugin_id and plugin_id not in rec_by_plugin:
            rec_by_plugin[plugin_id] = item
    exp_by_plugin: dict[str, dict[str, Any]] = {}
    for item in explanations:
        if not isinstance(item, dict):
            continue
        plugin_id = str(item.get("plugin_id") or "").strip()
        if plugin_id and plugin_id not in exp_by_plugin:
            exp_by_plugin[plugin_id] = item
    return rec_by_plugin, exp_by_plugin


def _class_for_plugin(plugin: dict[str, Any], taxonomy: dict[str, Any]) -> str:
    plugin_id = str(plugin.get("plugin_id") or "").strip()
    plugin_type = str(plugin.get("plugin_type") or "").strip().lower()
    overrides = taxonomy.get("plugin_overrides") if isinstance(taxonomy.get("plugin_overrides"), dict) else {}
    if plugin_id in overrides:
        return str(overrides[plugin_id]).strip()
    defaults = (
        taxonomy.get("plugin_type_default_class")
        if isinstance(taxonomy.get("plugin_type_default_class"), dict)
        else {}
    )
    if plugin_type in defaults:
        return str(defaults[plugin_type]).strip()
    return "supporting_signal_detectors"


def build_matrix(run_id: str, taxonomy_path: Path) -> dict[str, Any]:
    taxonomy = _read_yaml(taxonomy_path)
    classes = taxonomy.get("classes") if isinstance(taxonomy.get("classes"), dict) else {}
    manifests = _load_plugin_manifests()
    report = _report_for_run(run_id)
    rec_by_plugin, exp_by_plugin = _plugin_examples(report)
    report_plugins = (
        report.get("plugins") if isinstance(report.get("plugins"), dict) else {}
    )
    rows: list[dict[str, Any]] = []
    for plugin in manifests:
        plugin_id = str(plugin.get("plugin_id") or "").strip()
        class_id = _class_for_plugin(plugin, taxonomy)
        class_meta = classes.get(class_id) if isinstance(classes.get(class_id), dict) else {}
        expected_output_type = str(class_meta.get("expected_output_type") or "").strip()
        recommendation = rec_by_plugin.get(plugin_id)
        explanation = exp_by_plugin.get(plugin_id)
        plugin_payload = (
            report_plugins.get(plugin_id)
            if isinstance(report_plugins.get(plugin_id), dict)
            else None
        )
        plugin_status = (
            str(plugin_payload.get("status") or "").strip().lower()
            if isinstance(plugin_payload, dict)
            else ""
        )
        if isinstance(recommendation, dict):
            actionability_state = "actionable"
            reason_code = ""
            modeled_percent = recommendation.get("modeled_percent")
            modeled_supported = isinstance(modeled_percent, (int, float))
            example = {
                "kind": str(recommendation.get("kind") or ""),
                "action_type": str(recommendation.get("action_type") or recommendation.get("action") or ""),
                "recommendation": str(recommendation.get("recommendation") or ""),
                "modeled_percent": modeled_percent if modeled_supported else None,
            }
        elif isinstance(explanation, dict):
            actionability_state = "explained_na"
            reason_code = str(explanation.get("reason_code") or "").strip()
            if isinstance(plugin_payload, dict):
                findings = (
                    plugin_payload.get("findings")
                    if isinstance(plugin_payload.get("findings"), list)
                    else []
                )
                typed_findings = [item for item in findings if isinstance(item, dict)]
                blank_kind_count = int(
                    sum(1 for item in typed_findings if not str(item.get("kind") or "").strip())
                )
                derived_reason = derive_reason_code(
                    status=plugin_status or "unknown",
                    finding_count=int(len(typed_findings)),
                    blank_kind_count=blank_kind_count,
                    debug=plugin_payload.get("debug") if isinstance(plugin_payload.get("debug"), dict) else {},
                    findings=typed_findings,
                )
                if not reason_code or reason_code == "NOT_ROUTED_TO_ACTION":
                    reason_code = str(derived_reason or "").strip()
            if not reason_code:
                reason_code = "NON_DECISION_PLUGIN"
            modeled_supported = False
            example = {
                "kind": str(explanation.get("kind") or ""),
                "reason_code": reason_code,
                "plain_english_explanation": str(explanation.get("plain_english_explanation") or ""),
            }
        else:
            if not isinstance(plugin_payload, dict):
                actionability_state = "missing_output"
                reason_code = "NOT_IN_RUN_SCOPE"
                example = {}
            elif expected_output_type != "recommendation_items":
                actionability_state = "explained_na"
                reason_code = "NON_DECISION_PLUGIN"
                example = {
                    "kind": "non_actionable_explanation",
                    "reason_code": reason_code,
                    "plain_english_explanation": (
                        f"{plugin_id} is a non-decision plugin ({expected_output_type}) and is"
                        " tracked as explained N/A when recommendation lanes are absent."
                    ),
                }
            elif plugin_status in {"na", "not_applicable"}:
                actionability_state = "explained_na"
                reason_code = "NOT_APPLICABLE"
                example = {
                    "kind": "non_actionable_explanation",
                    "reason_code": reason_code,
                    "plain_english_explanation": (
                        f"{plugin_id} reported status '{plugin_status}' and is deterministically"
                        " classified as explained N/A."
                    ),
                }
            else:
                actionability_state = "missing_output"
                reason_code = "REPORT_SNAPSHOT_OMISSION"
                example = {
                    "kind": "missing_output",
                    "plugin_status": plugin_status or "unknown",
                }
            modeled_supported = False
        rows.append(
            {
                "plugin_id": plugin_id,
                "plugin_type": str(plugin.get("plugin_type") or ""),
                "plugin_name": str(plugin.get("name") or plugin_id),
                "plugin_class": class_id,
                "expected_output_type": expected_output_type,
                "actionability_state": actionability_state,
                "modeled_output_supported": bool(modeled_supported),
                "reason_code": reason_code,
                "plugin_status": plugin_status or None,
                "depends_on": plugin.get("depends_on") or [],
                "run_id": run_id,
                "example": example,
            }
        )

    class_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        class_id = str(row.get("plugin_class") or "")
        bucket = class_counts.setdefault(
            class_id,
            {"plugin_count": 0, "actionable_count": 0, "explained_na_count": 0, "missing_output_count": 0},
        )
        bucket["plugin_count"] += 1
        state = str(row.get("actionability_state") or "")
        if state == "actionable":
            bucket["actionable_count"] += 1
        elif state == "explained_na":
            bucket["explained_na_count"] += 1
        elif state == "missing_output":
            bucket["missing_output_count"] += 1

    return {
        "run_id": run_id,
        "taxonomy_path": str(taxonomy_path.relative_to(ROOT)),
        "plugin_count": len(rows),
        "class_counts": class_counts,
        "classes": classes,
        "plugins": sorted(rows, key=lambda row: str(row.get("plugin_id") or "")),
    }


def _to_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Plugin Class Actionability Matrix")
    lines.append("")
    lines.append("Generated by `scripts/build_plugin_class_actionability_matrix.py`.")
    lines.append("")
    lines.append(f"- run_id: `{str(payload.get('run_id') or '')}`")
    lines.append(f"- plugin_count: {int(payload.get('plugin_count') or 0)}")
    lines.append("")
    class_counts = payload.get("class_counts") if isinstance(payload.get("class_counts"), dict) else {}
    if class_counts:
        lines.append("## Class Summary")
        lines.append("")
        lines.append("| class | plugin_count | actionable | explained_na | missing_output |")
        lines.append("|---|---:|---:|---:|---:|")
        for class_id in sorted(class_counts.keys()):
            row = class_counts.get(class_id) if isinstance(class_counts.get(class_id), dict) else {}
            lines.append(
                f"| `{class_id}` | {int(row.get('plugin_count') or 0)} | {int(row.get('actionable_count') or 0)} | {int(row.get('explained_na_count') or 0)} | {int(row.get('missing_output_count') or 0)} |"
            )
        lines.append("")
    lines.append("## Plugin Matrix")
    lines.append("")
    lines.append("| plugin_id | type | class | state | expected_output | reason_code | example_kind |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in payload.get("plugins") if isinstance(payload.get("plugins"), list) else []:
        if not isinstance(row, dict):
            continue
        example = row.get("example") if isinstance(row.get("example"), dict) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{str(row.get('plugin_id') or '')}`",
                    str(row.get("plugin_type") or ""),
                    str(row.get("plugin_class") or ""),
                    str(row.get("actionability_state") or ""),
                    str(row.get("expected_output_type") or ""),
                    str(row.get("reason_code") or ""),
                    str(example.get("kind") or ""),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="")
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    parser.add_argument("--out-json", default=str(DEFAULT_JSON))
    parser.add_argument("--out-md", default=str(DEFAULT_MD))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    run_id = str(args.run_id).strip() or _latest_completed_run_id()
    taxonomy_path = Path(str(args.taxonomy)).resolve()
    payload = build_matrix(run_id, taxonomy_path)
    out_json = Path(str(args.out_json)).resolve()
    out_md = Path(str(args.out_md)).resolve()
    json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    md_text = _to_markdown(payload)

    unknown_class_plugins = [
        str(row.get("plugin_id") or "")
        for row in payload.get("plugins")
        if isinstance(row, dict)
        and str(row.get("plugin_class") or "")
        not in (payload.get("classes").keys() if isinstance(payload.get("classes"), dict) else [])
    ]
    if unknown_class_plugins:
        print(
            "unknown plugin_class assignments: " + ", ".join(sorted([p for p in unknown_class_plugins if p])),
            flush=True,
        )
        return 2

    if args.verify:
        if not out_json.exists() or not out_md.exists():
            return 2
        if out_json.read_text(encoding="utf-8") != json_text:
            return 2
        if out_md.read_text(encoding="utf-8") != md_text:
            return 2
        return 0

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json_text, encoding="utf-8")
    out_md.write_text(md_text, encoding="utf-8")
    print(f"run_id={run_id}")
    print(f"out_json={out_json}")
    print(f"out_md={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
