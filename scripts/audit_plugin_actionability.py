#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from statistic_harness.core.report import _build_recommendations
from statistic_harness.core.storage import Storage


ROOT = Path(__file__).resolve().parents[1]

_NEXT_STEP_ADAPTER_PREFIX = "Add or extend recommendation adapters"
_NEXT_STEP_DIRECT_ACTION_PREFIX = "Emit a direct-action finding kind"
_NEXT_STEP_PROCESS_TARGET_PREFIX = "Emit process-level targets"
_NEXT_STEP_SNAPSHOT = (
    "Plugin executed but is missing from report.plugins snapshot; include it in report serialization."
)
_NEXT_STEP_POLICY_PARENT_PREFIX = "Current target is policy-blocked"
_NEXT_STEP_CAPACITY_PREFIX = "Capacity impact was not applicable for current slices"
_NEXT_STEP_DOWNSTREAM_PREFIX = "Review downstream plugin outputs for action decisions:"
_NEXT_STEP_CONFIRM_PREFIX = "Confirm whether this plugin should have downstream consumers"
_NEXT_STEP_FAILURE_PREFIX = "Fix the plugin failure and rerun the full gauntlet."
_NEXT_STEP_PREREQ_PREFIX = "Verify input prerequisites;"
_NEXT_STEP_KIND_PREFIX = "Normalize finding kind values"
_NEXT_STEP_COVERAGE_PREFIX = "Confirm data coverage and add plugin-native findings"


def _classify_next_step_lane(row: dict[str, Any]) -> tuple[str, str]:
    state = str(row.get("actionability_state") or "").strip().lower()
    next_step = str(row.get("recommended_next_step") or "").strip()
    if state == "actionable":
        return "already_actionable", "actionable"
    if next_step.startswith(_NEXT_STEP_ADAPTER_PREFIX):
        return "adapter_extension", "actionable_or_deterministic_explained_non_actionable"
    if next_step.startswith(_NEXT_STEP_DIRECT_ACTION_PREFIX):
        return "direct_action_contract", "actionable_or_deterministic_explained_non_actionable"
    if next_step.startswith(_NEXT_STEP_PROCESS_TARGET_PREFIX):
        return "process_target_emission", "actionable_or_deterministic_explained_non_actionable"
    if next_step == _NEXT_STEP_SNAPSHOT:
        return "report_snapshot_serialization", "explained_non_actionable"
    if next_step.startswith(_NEXT_STEP_POLICY_PARENT_PREFIX):
        return "policy_parent_promotion", "actionable_or_deterministic_explained_non_actionable"
    if next_step.startswith(_NEXT_STEP_CAPACITY_PREFIX):
        return "capacity_slice_expansion", "actionable_or_deterministic_explained_non_actionable"
    if next_step.startswith(_NEXT_STEP_DOWNSTREAM_PREFIX):
        return "downstream_review_contract", "explained_non_actionable"
    if next_step.startswith(_NEXT_STEP_CONFIRM_PREFIX):
        return "standalone_or_downstream_decision", "explained_non_actionable"
    if next_step.startswith(_NEXT_STEP_FAILURE_PREFIX):
        return "failure_recovery_contract", "actionable_or_deterministic_explained_non_actionable"
    if next_step.startswith(_NEXT_STEP_PREREQ_PREFIX):
        return "prerequisite_contract", "actionable_or_deterministic_explained_non_actionable"
    if next_step.startswith(_NEXT_STEP_KIND_PREFIX):
        return "finding_schema_contract", "actionable_or_deterministic_explained_non_actionable"
    if next_step.startswith(_NEXT_STEP_COVERAGE_PREFIX):
        return "data_coverage_contract", "actionable_or_deterministic_explained_non_actionable"
    if not next_step:
        return "missing_next_step", "manual_triage_required"
    return "unmapped_next_step", "manual_triage_required"


def _build_next_step_work_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cluster_counts: Counter[str] = Counter()
    state_transition_counts: Counter[str] = Counter()
    unmapped_plugins: list[str] = []
    blank_non_actionable_plugins: list[str] = []
    pending_plugins: list[str] = []
    contract_rows: list[dict[str, Any]] = []

    for row in rows:
        plugin_id = str(row.get("plugin_id") or "").strip()
        lane_id, expected_post_state = _classify_next_step_lane(row)
        cluster_counts[lane_id] += 1
        current_state = str(row.get("actionability_state") or "").strip().lower()
        reason_code = str(row.get("reason_code") or "").strip()
        state_transition_counts[f"{reason_code or 'UNSPECIFIED'}->{lane_id}->{expected_post_state}"] += 1
        recommended_next_step = str(row.get("recommended_next_step") or "").strip()
        if current_state != "actionable":
            pending_plugins.append(plugin_id)
        if current_state != "actionable" and not recommended_next_step:
            blank_non_actionable_plugins.append(plugin_id)
        if lane_id in {"missing_next_step", "unmapped_next_step"} and current_state != "actionable":
            unmapped_plugins.append(plugin_id)

        contract_rows.append(
            {
                "plugin_id": plugin_id,
                "current_reason_code": reason_code or None,
                "current_actionability_state": current_state or None,
                "recommended_next_step": recommended_next_step or None,
                "next_step_lane_id": lane_id,
                "expected_post_state": expected_post_state,
                "implementation_status": (
                    "already_actionable" if current_state == "actionable" else "pending_implementation"
                ),
            }
        )

    return {
        "plugin_count": int(len(rows)),
        "cluster_counts": {k: int(v) for k, v in sorted(cluster_counts.items())},
        "state_transition_counts": {k: int(v) for k, v in sorted(state_transition_counts.items())},
        "unmapped_next_step_plugins": sorted(p for p in unmapped_plugins if p),
        "blank_non_actionable_next_step_plugins": sorted(
            p for p in blank_non_actionable_plugins if p
        ),
        "pending_plugin_ids": sorted(p for p in pending_plugins if p),
        "rows": contract_rows,
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to load JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return payload


def _manifest_index() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
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
        out[plugin_id] = {
            "plugin_type": str(payload.get("type") or "").strip().lower(),
            "name": str(payload.get("name") or plugin_id).strip(),
        }
    return out


def _top_kinds(findings: list[dict[str, Any]]) -> list[str]:
    counts = Counter(
        str(item.get("kind") or "").strip()
        for item in findings
        if str(item.get("kind") or "").strip()
    )
    return [kind for kind, _ in counts.most_common(6)]


def _executed_plugin_rows(run_id: str) -> dict[str, dict[str, Any]]:
    db_path = ROOT / "appdata" / "state.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        exec_rows = conn.execute(
            """
            SELECT plugin_id, status
            FROM plugin_executions
            WHERE run_id = ?
            ORDER BY execution_id ASC
            """,
            (run_id,),
        ).fetchall()
        result_rows = conn.execute(
            """
            SELECT plugin_id, status, summary
            FROM plugin_results_v2
            WHERE run_id = ?
            ORDER BY result_id ASC
            """,
            (run_id,),
        ).fetchall()
    finally:
        conn.close()
    out: dict[str, dict[str, Any]] = {}
    for plugin_id, status in exec_rows:
        pid = str(plugin_id or "").strip()
        if not pid:
            continue
        bucket = out.setdefault(pid, {})
        bucket["execution_status"] = str(status or "").strip().lower()
    for plugin_id, status, summary in result_rows:
        pid = str(plugin_id or "").strip()
        if not pid:
            continue
        bucket = out.setdefault(pid, {})
        bucket["result_status"] = str(status or "").strip().lower()
        bucket["result_summary"] = str(summary or "").strip()
    return out


def _recommendations_for_run(
    report: dict[str, Any],
    *,
    run_id: str,
    recompute: bool,
) -> dict[str, Any]:
    if not recompute:
        recs = report.get("recommendations")
        return recs if isinstance(recs, dict) else {}
    run_dir = ROOT / "appdata" / "runs" / run_id
    db_path = ROOT / "appdata" / "state.sqlite"
    # Full-context recompute is the deterministic source of truth for actionability
    # because modeled deltas and close-window context can depend on storage/run artifacts.
    try:
        storage = Storage(db_path, tenant_id=None, mode="ro", initialize=False)
        return _build_recommendations(report, storage, run_dir=run_dir)
    except Exception:
        # Fallback keeps audit operational even if state DB or run_dir context is unavailable.
        return _build_recommendations(report, storage=None, run_dir=None)


def audit_run(run_id: str, *, recompute: bool) -> dict[str, Any]:
    run_dir = ROOT / "appdata" / "runs" / run_id
    report_path = run_dir / "report.json"
    if not report_path.exists():
        raise SystemExit(f"Missing report for run_id={run_id}: {report_path}")
    report = _load_json(report_path)
    recommendations = _recommendations_for_run(report, run_id=run_id, recompute=recompute)
    manifests = _manifest_index()
    executed = _executed_plugin_rows(run_id)
    report_plugins = report.get("plugins") if isinstance(report.get("plugins"), dict) else {}
    recommendation_items = (
        recommendations.get("items") if isinstance(recommendations.get("items"), list) else []
    )
    discovery = recommendations.get("discovery") if isinstance(recommendations.get("discovery"), dict) else {}
    discovery_actionable_ids = {
        str(pid).strip()
        for pid in (discovery.get("actionable_plugin_ids_all") or [])
        if isinstance(pid, str) and str(pid).strip()
    }
    explanation_items = (
        (recommendations.get("explanations") or {}).get("items")
        if isinstance(recommendations.get("explanations"), dict)
        else []
    )
    rec_count_by_plugin: dict[str, int] = defaultdict(int)
    for item in recommendation_items:
        if not isinstance(item, dict):
            continue
        plugin_id = str(item.get("plugin_id") or "").strip()
        if plugin_id:
            rec_count_by_plugin[plugin_id] += 1
    explanation_by_plugin: dict[str, dict[str, Any]] = {}
    for item in explanation_items:
        if not isinstance(item, dict):
            continue
        plugin_id = str(item.get("plugin_id") or "").strip()
        if plugin_id and plugin_id not in explanation_by_plugin:
            explanation_by_plugin[plugin_id] = item

    plugin_ids = sorted(
        set(executed.keys()) | set(str(v).strip() for v in report_plugins.keys() if str(v).strip())
    )
    rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    missing_plugins: list[str] = []
    for plugin_id in plugin_ids:
        manifest = manifests.get(plugin_id) if isinstance(manifests.get(plugin_id), dict) else {}
        plugin_payload = (
            report_plugins.get(plugin_id) if isinstance(report_plugins.get(plugin_id), dict) else {}
        )
        executed_meta = executed.get(plugin_id) if isinstance(executed.get(plugin_id), dict) else {}
        findings = plugin_payload.get("findings") if isinstance(plugin_payload.get("findings"), list) else []
        typed_findings = [item for item in findings if isinstance(item, dict)]
        rec_count = int(rec_count_by_plugin.get(plugin_id, 0))
        actionable_via_discovery = plugin_id in discovery_actionable_ids
        explanation = explanation_by_plugin.get(plugin_id) if isinstance(explanation_by_plugin.get(plugin_id), dict) else {}
        reason_code = str(explanation.get("reason_code") or "").strip()
        reason_detail = str(explanation.get("reason_code_detail") or "").strip()
        if rec_count > 0 or actionable_via_discovery:
            state = "actionable"
            reason_code = ""
            reason_detail = ""
            explanation = {}
        elif explanation:
            state = "explained_non_actionable"
        elif executed_meta:
            state = "explained_non_actionable"
            reason_code = "REPORT_SNAPSHOT_OMISSION"
            reason_detail = None
            explanation = {
                "recommended_next_step": (
                    "Plugin executed but is missing from report.plugins snapshot; include it in report serialization."
                )
            }
        else:
            state = "missing_output"
            missing_plugins.append(plugin_id)
        state_counts[state] += 1
        if reason_code:
            reason_counts[reason_code] += 1
        rows.append(
            {
                "plugin_id": plugin_id,
                "plugin_type": str(manifest.get("plugin_type") or "").strip(),
                "plugin_name": str(manifest.get("name") or plugin_id).strip(),
                "plugin_status": (
                    str(plugin_payload.get("status") or "").strip().lower()
                    or str(executed_meta.get("result_status") or executed_meta.get("execution_status") or "").strip().lower()
                    or "missing"
                ),
                "actionability_state": state,
                "recommendation_count": rec_count,
                "actionable_via_discovery": bool(actionable_via_discovery),
                "reason_code": reason_code or None,
                "reason_code_detail": reason_detail or None,
                "recommended_next_step": str(explanation.get("recommended_next_step") or "").strip() or None,
                "finding_count": int(len(typed_findings)),
                "finding_kind_preview": _top_kinds(typed_findings),
                "plugin_summary": (
                    str(plugin_payload.get("summary") or "").strip()
                    or str(executed_meta.get("result_summary") or "").strip()
                    or None
                ),
            }
        )
    work_contract = _build_next_step_work_contract(rows)
    lane_by_plugin = {
        str(item.get("plugin_id") or "").strip(): item
        for item in (work_contract.get("rows") or [])
        if isinstance(item, dict) and str(item.get("plugin_id") or "").strip()
    }
    for row in rows:
        pid = str(row.get("plugin_id") or "").strip()
        if not pid:
            continue
        lane_row = lane_by_plugin.get(pid)
        if isinstance(lane_row, dict):
            row["next_step_lane_id"] = str(lane_row.get("next_step_lane_id") or "").strip() or None
            row["expected_post_state"] = str(lane_row.get("expected_post_state") or "").strip() or None

    return {
        "run_id": run_id,
        "recomputed_recommendations": bool(recompute),
        "plugin_count": int(len(rows)),
        "state_counts": {k: int(v) for k, v in sorted(state_counts.items())},
        "reason_code_counts": {k: int(v) for k, v in sorted(reason_counts.items())},
        "missing_output_plugins": sorted(missing_plugins),
        "next_step_work_contract": work_contract,
        "plugins": rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "plugin_id",
        "plugin_type",
        "plugin_name",
        "plugin_status",
        "actionability_state",
        "recommendation_count",
        "actionable_via_discovery",
        "reason_code",
        "reason_code_detail",
        "recommended_next_step",
        "next_step_lane_id",
        "expected_post_state",
        "finding_count",
        "finding_kind_preview",
        "plugin_summary",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["finding_kind_preview"] = ";".join(
                str(v).strip() for v in (row.get("finding_kind_preview") or []) if str(v).strip()
            )
            writer.writerow(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--recompute-recommendations", action="store_true")
    parser.add_argument("--strict-no-non-decision", action="store_true")
    parser.add_argument("--strict-no-missing-output", action="store_true")
    parser.add_argument("--strict-next-step-covered", action="store_true")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-csv", default="")
    args = parser.parse_args()

    run_id = str(args.run_id).strip()
    payload = audit_run(run_id, recompute=bool(args.recompute_recommendations))
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out_json = str(args.out_json).strip()
    out_csv = str(args.out_csv).strip()
    if out_json:
        out_path = Path(out_json)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
    if out_csv:
        out_path = Path(out_csv)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        _write_csv(out_path, payload.get("plugins") if isinstance(payload.get("plugins"), list) else [])

    print(rendered, end="")

    if bool(args.strict_no_non_decision):
        if int((payload.get("reason_code_counts") or {}).get("NON_DECISION_PLUGIN", 0)) > 0:
            return 2
    if bool(args.strict_no_missing_output):
        if int(len(payload.get("missing_output_plugins") or [])) > 0:
            return 3
    if bool(args.strict_next_step_covered):
        contract = payload.get("next_step_work_contract")
        if not isinstance(contract, dict):
            return 4
        unmapped = contract.get("unmapped_next_step_plugins")
        blank_non_actionable = contract.get("blank_non_actionable_next_step_plugins")
        if int(len(unmapped or [])) > 0 or int(len(blank_non_actionable or [])) > 0:
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
