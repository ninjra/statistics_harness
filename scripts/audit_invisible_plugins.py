#!/usr/bin/env python3
"""Audit invisible plugins: classify non-contributing plugins as correctly-silent vs incorrectly-filtered.

Reads a report.json (or loads from SQLite) and produces a structured audit of all
plugins that produced findings but did not contribute any recommendations.

Usage:
    python scripts/audit_invisible_plugins.py --report-json path/to/report.json
    python scripts/audit_invisible_plugins.py --run-id <run_id>
    python scripts/audit_invisible_plugins.py --report-json path/to/report.json --out-dir audit_output/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
PLUGIN_KIND_MAP_PATH = CONFIG_DIR / "plugin_kind_map.yaml"


def _load_plugin_kind_map() -> dict[str, str]:
    if not PLUGIN_KIND_MAP_PATH.exists():
        return {}
    with open(PLUGIN_KIND_MAP_PATH) as fh:
        data = yaml.safe_load(fh) or {}
    return dict(data.get("mappings") or {})


def _load_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.report_json:
        with open(args.report_json) as fh:
            return json.load(fh)
    if args.run_id:
        import sqlite3

        db_path = ROOT / "appdata" / "state.sqlite"
        if not db_path.exists():
            print(f"ERROR: SQLite database not found at {db_path}", file=sys.stderr)
            sys.exit(1)
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT report_json FROM runs WHERE run_id = ? LIMIT 1",
            (args.run_id,),
        ).fetchone()
        conn.close()
        if not row:
            print(f"ERROR: run_id '{args.run_id}' not found in state.sqlite", file=sys.stderr)
            sys.exit(1)
        return json.loads(row[0])
    print("ERROR: Provide --report-json or --run-id", file=sys.stderr)
    sys.exit(1)


def _extract_non_actionable_explanations(report: dict[str, Any]) -> list[dict[str, Any]]:
    recs = report.get("recommendations") if isinstance(report.get("recommendations"), dict) else {}
    explanations = recs.get("explanations") if isinstance(recs.get("explanations"), dict) else {}
    items = explanations.get("items") if isinstance(explanations.get("items"), list) else []
    return [item for item in items if isinstance(item, dict)]


def _extract_contributing_plugin_ids(report: dict[str, Any]) -> set[str]:
    recs = report.get("recommendations") if isinstance(report.get("recommendations"), dict) else {}
    items = recs.get("items") if isinstance(recs.get("items"), list) else []
    return {
        str(row.get("plugin_id") or "").strip()
        for row in items
        if isinstance(row, dict) and str(row.get("plugin_id") or "").strip()
    }


def _extract_plugin_findings(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    plugins = report.get("plugins") if isinstance(report.get("plugins"), dict) else {}
    result: dict[str, list[dict[str, Any]]] = {}
    for pid, plugin in plugins.items():
        if not isinstance(plugin, dict):
            continue
        findings = [f for f in (plugin.get("findings") or []) if isinstance(f, dict)]
        if findings:
            result[str(pid)] = findings
    return result


def _finding_kinds_for_plugin(findings: list[dict[str, Any]]) -> set[str]:
    return {
        str(f.get("kind") or "").strip().lower()
        for f in findings
        if str(f.get("kind") or "").strip()
    }


def _has_process_targeting(findings: list[dict[str, Any]]) -> bool:
    for f in findings:
        for key in ("process_norm", "process", "process_id"):
            val = str(f.get(key) or "").strip().lower()
            if val and val not in {"", "all", "any", "global", "multiple", "(multiple)"}:
                return True
        evidence = f.get("evidence") if isinstance(f.get("evidence"), dict) else {}
        for key in ("process_norm", "process", "process_id"):
            val = str(evidence.get(key) or "").strip().lower()
            if val and val not in {"", "all", "any", "global", "multiple", "(multiple)"}:
                return True
    return False


def _max_confidence(findings: list[dict[str, Any]]) -> float:
    best = 0.0
    for f in findings:
        conf = f.get("confidence")
        if isinstance(conf, (int, float)):
            best = max(best, float(conf))
    return best


def _suspect_false_negative(
    plugin_id: str,
    reason_code: str,
    findings: list[dict[str, Any]],
    kind_map: dict[str, str],
    generic_kinds: set[str],
    backstop_kinds: set[str],
) -> dict[str, Any] | None:
    finding_count = len(findings)
    max_conf = _max_confidence(findings)
    has_process = _has_process_targeting(findings)
    finding_kinds = _finding_kinds_for_plugin(findings)

    suspect_reasons: list[str] = []

    if reason_code == "OBSERVATION_ONLY" and finding_count > 0 and max_conf > 0.5:
        suspect_reasons.append(
            f"High confidence ({max_conf:.2f}) with {finding_count} findings marked OBSERVATION_ONLY"
        )

    if reason_code == "OBSERVATION_ONLY" and has_process:
        suspect_reasons.append("Has process-specific findings but marked OBSERVATION_ONLY")

    if reason_code == "NO_DECISION_SIGNAL" and has_process and max_conf > 0.3:
        suspect_reasons.append(
            f"Has process targets and confidence={max_conf:.2f} but marked NO_DECISION_SIGNAL"
        )

    if reason_code == "ADAPTER_RULE_MISSING":
        mapped_kind = kind_map.get(plugin_id, "")
        if mapped_kind and mapped_kind not in generic_kinds and mapped_kind not in backstop_kinds:
            suspect_reasons.append(
                f"Plugin kind '{mapped_kind}' is in kind_map but missing from routing maps"
            )
        for fk in finding_kinds:
            if fk and fk not in generic_kinds and fk not in backstop_kinds:
                suspect_reasons.append(f"Finding kind '{fk}' has no routing rule")

    if not suspect_reasons:
        return None

    return {
        "plugin_id": plugin_id,
        "reason_code": reason_code,
        "finding_count": finding_count,
        "max_confidence": max_conf,
        "has_process_targeting": has_process,
        "finding_kinds": sorted(finding_kinds),
        "suspect_reasons": suspect_reasons,
        "priority_score": finding_count * max_conf,
    }


def run_audit(report: dict[str, Any]) -> dict[str, Any]:
    kind_map = _load_plugin_kind_map()
    contributing_ids = _extract_contributing_plugin_ids(report)
    plugin_findings = _extract_plugin_findings(report)
    non_actionable_items = _extract_non_actionable_explanations(report)

    # Known routing map kinds (from report.py)
    generic_kinds = {
        "anomaly", "changepoint", "chi_square_association", "cluster",
        "process_variant", "tail_isolation", "sequence_classification",
        "tda_betti_curve_changepoint", "bayesian_point_displacement",
        "chain_makespan", "percentile_stats", "dependence_shift", "graph_edge",
        "ideaspace_gap", "verified_route_action_plan", "correlation", "distribution",
        "time_series", "survival", "regression", "graph", "causal", "counterfactual",
        "recommendation", "role_inference", "close_cycle_capacity_model",
        "capacity_scale_model",
    }
    backstop_kinds = generic_kinds | {
        "tda_betti_curve_changepoint", "bayesian_point_displacement",
    }

    # Group non-actionable explanations by reason code
    by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in non_actionable_items:
        reason = str(item.get("reason_code") or item.get("reason") or "UNKNOWN").strip()
        by_reason[reason].append(item)

    reason_summary: dict[str, int] = {
        reason: len(items) for reason, items in sorted(by_reason.items(), key=lambda kv: -len(kv[1]))
    }

    # Find plugins with findings that are not contributing
    invisible_plugins_with_findings: dict[str, list[dict[str, Any]]] = {}
    for pid, findings in plugin_findings.items():
        if pid not in contributing_ids:
            invisible_plugins_with_findings[pid] = findings

    # Build suspect list
    suspects: list[dict[str, Any]] = []
    for item in non_actionable_items:
        pid = str(item.get("plugin_id") or "").strip()
        reason = str(item.get("reason_code") or item.get("reason") or "UNKNOWN").strip()
        findings = plugin_findings.get(pid, [])
        if not findings:
            continue
        suspect = _suspect_false_negative(pid, reason, findings, kind_map, generic_kinds, backstop_kinds)
        if suspect:
            suspects.append(suspect)

    suspects.sort(key=lambda s: -s["priority_score"])

    # ADAPTER_RULE_MISSING detail
    adapter_missing: list[dict[str, Any]] = []
    for item in by_reason.get("ADAPTER_RULE_MISSING", []):
        pid = str(item.get("plugin_id") or "").strip()
        findings = plugin_findings.get(pid, [])
        finding_kinds = _finding_kinds_for_plugin(findings)
        mapped_kind = kind_map.get(pid, "")
        adapter_missing.append({
            "plugin_id": pid,
            "mapped_kind": mapped_kind,
            "finding_kinds": sorted(finding_kinds),
            "finding_count": len(findings),
            "has_process_targeting": _has_process_targeting(findings),
            "max_confidence": _max_confidence(findings),
        })

    # OBSERVATION_ONLY with process targeting
    obs_only_upgradeable: list[dict[str, Any]] = []
    for item in by_reason.get("OBSERVATION_ONLY", []):
        pid = str(item.get("plugin_id") or "").strip()
        findings = plugin_findings.get(pid, [])
        if _has_process_targeting(findings):
            obs_only_upgradeable.append({
                "plugin_id": pid,
                "finding_kinds": sorted(_finding_kinds_for_plugin(findings)),
                "finding_count": len(findings),
                "max_confidence": _max_confidence(findings),
            })

    return {
        "schema_version": "v1",
        "summary": {
            "total_plugins_with_findings": len(plugin_findings),
            "contributing_plugins": len(contributing_ids),
            "invisible_plugins_with_findings": len(invisible_plugins_with_findings),
            "total_non_actionable_explanations": len(non_actionable_items),
            "suspect_false_negatives": len(suspects),
        },
        "reason_code_summary": reason_summary,
        "suspect_false_negatives": suspects,
        "adapter_rule_missing_detail": adapter_missing,
        "observation_only_upgradeable": obs_only_upgradeable,
    }


def _render_markdown(audit: dict[str, Any]) -> str:
    lines: list[str] = []
    summary = audit["summary"]
    lines.append("# Invisible Plugin Audit\n")
    lines.append(f"- **Total plugins with findings**: {summary['total_plugins_with_findings']}")
    lines.append(f"- **Contributing plugins**: {summary['contributing_plugins']}")
    lines.append(f"- **Invisible (with findings)**: {summary['invisible_plugins_with_findings']}")
    lines.append(
        f"- **Total non-actionable explanations**: {summary['total_non_actionable_explanations']}"
    )
    lines.append(f"- **Suspect false negatives**: {summary['suspect_false_negatives']}")
    lines.append("")

    lines.append("## Reason Code Summary\n")
    lines.append("| Reason Code | Count |")
    lines.append("|---|---|")
    for reason, count in audit["reason_code_summary"].items():
        lines.append(f"| {reason} | {count} |")
    lines.append("")

    if audit["suspect_false_negatives"]:
        lines.append("## Suspect False Negatives\n")
        lines.append("| Plugin ID | Reason | Findings | Confidence | Process? | Priority |")
        lines.append("|---|---|---|---|---|---|")
        for s in audit["suspect_false_negatives"][:30]:
            lines.append(
                f"| {s['plugin_id']} | {s['reason_code']} | {s['finding_count']} "
                f"| {s['max_confidence']:.2f} | {'Y' if s['has_process_targeting'] else 'N'} "
                f"| {s['priority_score']:.1f} |"
            )
        lines.append("")

    if audit["adapter_rule_missing_detail"]:
        lines.append("## ADAPTER_RULE_MISSING Detail\n")
        lines.append("| Plugin ID | Mapped Kind | Finding Kinds | Count |")
        lines.append("|---|---|---|---|")
        for a in audit["adapter_rule_missing_detail"]:
            lines.append(
                f"| {a['plugin_id']} | {a['mapped_kind']} | {', '.join(a['finding_kinds'])} "
                f"| {a['finding_count']} |"
            )
        lines.append("")

    if audit["observation_only_upgradeable"]:
        lines.append("## OBSERVATION_ONLY With Process Targeting (Upgrade Candidates)\n")
        lines.append("| Plugin ID | Finding Kinds | Count | Confidence |")
        lines.append("|---|---|---|---|")
        for o in audit["observation_only_upgradeable"]:
            lines.append(
                f"| {o['plugin_id']} | {', '.join(o['finding_kinds'])} "
                f"| {o['finding_count']} | {o['max_confidence']:.2f} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit invisible plugins")
    parser.add_argument("--report-json", type=Path, help="Path to report.json")
    parser.add_argument("--run-id", type=str, help="Run ID to load from SQLite")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "docs" / "release_evidence",
        help="Output directory for audit files",
    )
    args = parser.parse_args()

    report = _load_report(args)
    audit = run_audit(report)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "invisible_plugin_audit.json"
    md_path = out_dir / "invisible_plugin_audit.md"

    with open(json_path, "w") as fh:
        json.dump(audit, fh, indent=2)
    with open(md_path, "w") as fh:
        fh.write(_render_markdown(audit))

    print(f"Audit written to:")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    print()
    print(f"Summary:")
    for key, value in audit["summary"].items():
        print(f"  {key}: {value}")
    print()
    print(f"Top reason codes:")
    for reason, count in list(audit["reason_code_summary"].items())[:10]:
        print(f"  {reason}: {count}")
    if audit["suspect_false_negatives"]:
        print()
        print(f"Top 5 suspect false negatives:")
        for s in audit["suspect_false_negatives"][:5]:
            print(f"  {s['plugin_id']}: {s['reason_code']} ({', '.join(s['suspect_reasons'][:1])})")


if __name__ == "__main__":
    main()
