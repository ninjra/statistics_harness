from __future__ import annotations

from typing import Any


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
        return code
    for item in findings or []:
        if not isinstance(item, dict):
            continue
        candidate = str(item.get("reason_code") or "").strip()
        if candidate:
            return candidate
    status_norm = str(status or "").strip().lower()
    if status_norm == "error":
        return "PLUGIN_ERROR"
    if status_norm == "na":
        return "NOT_APPLICABLE"
    if int(finding_count or 0) <= 0:
        return "NO_FINDINGS"
    if int(blank_kind_count or 0) > 0:
        return "FINDING_KIND_MISSING"
    return "NOT_ROUTED_TO_ACTION"


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
            f"{pid} is not directly actionable in this run because it was marked not applicable: "
            f"{summary or 'no applicable input'}."
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
