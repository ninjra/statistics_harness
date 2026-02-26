from __future__ import annotations

from typing import Any

NON_ADJUSTABLE_PROCESSES = {
    "losextchld",
    "losloadcld",
    "jbcreateje",
    "jboachild",
    "jbvalcdblk",
    "jbinvoice",
    "postwkfl",
    "jbpreproof",
    "rdimpairje",
}

_REASON_CODE_NORMALIZATION = {
    "NOT_APPLICABLE": "PREREQUISITE_UNMET",
    "CAPACITY_IMPACT_NOT_APPLICABLE": "CAPACITY_IMPACT_CONSTRAINT",
    "NO_ACTIONABLE_RESULT": "NO_DECISION_SIGNAL",
    "NO_ROUTING_RULE_MATCH": "ADAPTER_RULE_MISSING",
}


def _normalize_reason_code(code: str) -> str:
    key = str(code or "").strip()
    if not key:
        return ""
    return str(_REASON_CODE_NORMALIZATION.get(key, key))


def derive_reason_code(
    *,
    status: str,
    finding_count: int,
    blank_kind_count: int,
    debug: dict[str, Any] | None,
    findings: list[dict[str, Any]] | None,
) -> str:
    debug_map = debug if isinstance(debug, dict) else {}
    code = str(debug_map.get("reason_code") or "").strip()
    if code:
        return _normalize_reason_code(code)
    typed_findings = [item for item in (findings or []) if isinstance(item, dict)]
    for item in typed_findings:
        if not isinstance(item, dict):
            continue
        candidate = str(item.get("reason_code") or "").strip()
        if candidate:
            return _normalize_reason_code(candidate)
    kinds = {str(item.get("kind") or "").strip() for item in typed_findings}
    if "close_cycle_capacity_impact" in kinds:
        if all(
            str(item.get("decision") or "").strip().lower() == "not_applicable"
            for item in typed_findings
            if str(item.get("kind") or "").strip() == "close_cycle_capacity_impact"
        ):
            return "CAPACITY_IMPACT_CONSTRAINT"
    if "close_cycle_capacity_model" in kinds:
        modeled = [
            item
            for item in typed_findings
            if str(item.get("kind") or "").strip() == "close_cycle_capacity_model"
            and str(item.get("decision") or "").strip().lower() == "modeled"
        ]
        if modeled and all(
            isinstance(item.get("baseline_value"), (int, float))
            and isinstance(item.get("modeled_value"), (int, float))
            and float(item.get("modeled_value")) >= float(item.get("baseline_value"))
            for item in modeled
        ):
            return "NO_MODELED_CAPACITY_GAIN"
    if "close_cycle_revenue_compression" in kinds:
        modeled = [
            item
            for item in typed_findings
            if str(item.get("kind") or "").strip() == "close_cycle_revenue_compression"
            and str(item.get("decision") or "").strip().lower() == "modeled"
        ]
        if modeled and all(
            isinstance(item.get("baseline_value"), (int, float))
            and isinstance(item.get("modeled_value"), (int, float))
            and float(item.get("modeled_value")) >= float(item.get("baseline_value"))
            for item in modeled
        ):
            return "NO_REVENUE_COMPRESSION_PRESSURE"
    if "close_cycle_share_shift" in kinds and all(
        not isinstance(item.get("share_delta"), (int, float)) or float(item.get("share_delta")) <= 0.0
        for item in typed_findings
        if str(item.get("kind") or "").strip() == "close_cycle_share_shift"
    ):
        return "SHARE_SHIFT_BELOW_THRESHOLD"
    if "plugin_not_applicable" in kinds:
        return "NO_DECISION_SIGNAL"
    if "plugin_observation" in kinds:
        return "NO_DECISION_SIGNAL"
    if "actionable_ops_lever" in kinds:
        return "ADAPTER_RULE_MISSING"
    share_shift_rows = [
        item
        for item in typed_findings
        if str(item.get("kind") or "").strip() == "close_cycle_share_shift"
    ]
    if share_shift_rows:
        process_norms = {
            str(item.get("process_norm") or item.get("process") or "").strip().lower()
            for item in share_shift_rows
            if str(item.get("process_norm") or item.get("process") or "").strip()
        }
        if process_norms and all(proc in NON_ADJUSTABLE_PROCESSES for proc in process_norms):
            return "EXCLUDED_BY_PROCESS_POLICY"
    status_norm = str(status or "").strip().lower()
    if status_norm == "error":
        return "PLUGIN_ERROR"
    if status_norm == "na":
        return "PREREQUISITE_UNMET"
    if int(finding_count or 0) <= 0:
        return "NO_FINDINGS"
    if int(blank_kind_count or 0) > 0:
        return "FINDING_KIND_MISSING"
    return "ADAPTER_RULE_MISSING"


def plain_english_explanation(
    *,
    plugin_id: str,
    plugin_type: str,
    status: str,
    summary: str,
    finding_count: int,
    blank_kind_count: int,
    downstream_plugins: list[str] | None,
) -> str:
    pid = str(plugin_id or "").strip() or "unknown_plugin"
    ptype = str(plugin_type or "").strip().lower()
    status_norm = str(status or "").strip().lower()
    text = ""
    if status_norm == "error":
        text = (
            f"{pid} did not produce an actionable recommendation because it failed in this run: "
            f"{summary or 'execution error'}."
        )
    elif status_norm == "na":
        text = (
            f"{pid} is not directly actionable in this run because required prerequisites were not met: "
            f"{summary or 'missing required input/coverage'}."
        )
    elif int(finding_count or 0) <= 0:
        text = f"{pid} completed but did not emit findings that can drive an action."
    elif int(blank_kind_count or 0) > 0:
        text = (
            f"{pid} produced {int(finding_count)} finding(s), but {int(blank_kind_count)} lack a finding kind, "
            "so action routing cannot classify them."
        )
    else:
        text = (
            f"{pid} produced {int(finding_count)} finding(s), but none matched current action-routing rules."
        )
    downstream = [str(v).strip() for v in (downstream_plugins or []) if str(v).strip()]
    if ptype and ptype != "analysis":
        if downstream:
            sample = ", ".join(downstream[:12])
            suffix = f" (+{len(downstream) - 12} more)" if len(downstream) > 12 else ""
            text += f" This is a {ptype} plugin; its output feeds downstream plugins: {sample}{suffix}."
        else:
            text += f" This is a {ptype} plugin with no downstream plugin dependencies in this run."
    return text


def recommended_next_step(
    *,
    plugin_type: str,
    status: str,
    finding_count: int,
    blank_kind_count: int,
    downstream_plugins: list[str] | None,
) -> str:
    ptype = str(plugin_type or "").strip().lower()
    status_norm = str(status or "").strip().lower()
    downstream = [str(v).strip() for v in (downstream_plugins or []) if str(v).strip()]
    if ptype and ptype != "analysis":
        if downstream:
            return (
                "Review downstream plugin outputs for action decisions: "
                + ", ".join(downstream[:5])
                + ("." if len(downstream) <= 5 else ", ...")
            )
        return "Confirm whether this plugin should have downstream consumers or remain standalone."
    if status_norm == "error":
        return "Fix the plugin failure and rerun the full gauntlet."
    if status_norm == "na":
        return "Verify input prerequisites; if this should apply, update plugin gating and rerun."
    if int(blank_kind_count or 0) > 0:
        return "Normalize finding kind values for this plugin so routing can map outputs to actions."
    if int(finding_count or 0) <= 0:
        return (
            "Confirm data coverage and add plugin-native findings or explanation mapping so this plugin "
            "contributes decision support."
        )
    return "Add or extend recommendation adapters for this plugin's finding families."
