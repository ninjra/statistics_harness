from __future__ import annotations

import math
from typing import Any

from .action_types import _action_type_obviousness, _tier_score_for_item
from .config import _include_capacity_recommendations, _ranking_version
from .process_targeting import _recommendation_process_hint
from .text import _format_issue_value
from ..stat_controls import confidence_from_p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_num(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _item_process_norm(item: dict[str, Any]) -> str:
    where = item.get("where") if isinstance(item.get("where"), dict) else {}
    for key in ("process_norm", "process", "process_id", "activity"):
        value = where.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    target = item.get("target")
    if isinstance(target, str) and target.strip():
        return target.strip().lower()
    return ""


# ---------------------------------------------------------------------------
# Scoring & sort keys
# ---------------------------------------------------------------------------


def _modeled_improvement_percent(baseline: Any, modeled: Any) -> float | None:
    if not isinstance(baseline, (int, float)) or not isinstance(modeled, (int, float)):
        return None
    base = float(baseline)
    mod = float(modeled)
    if base <= 0.0:
        return None
    return ((base - mod) / base) * 100.0


def _discovery_recommendation_sort_key(item: dict[str, Any]) -> tuple[int, int, float, float, float]:
    # Keep existing coarse family ordering, but prefer Tier 1 structural actions within a family.
    plugin_id = str(item.get("plugin_id") or "")
    kind = str(item.get("kind") or "")
    action_type = str(item.get("action_type") or item.get("action") or "")

    priority = 0
    if plugin_id == "analysis_actionable_ops_levers_v1" or kind == "actionable_ops_lever":
        priority = 6
    elif plugin_id.startswith("analysis_ideaspace_"):
        priority = 5
    elif "sequence" in plugin_id or "bottleneck" in plugin_id or "conformance" in plugin_id:
        priority = 4
    elif plugin_id == "analysis_upload_linkage":
        priority = 3
    elif action_type and action_type not in ("review", "tune_threshold"):
        priority = 2
    elif plugin_id in ("analysis_queue_delay_decomposition", "analysis_busy_period_segmentation_v2"):
        priority = 1

    try:
        score = float(item.get("relevance_score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    try:
        impact = float(item.get("impact_hours") or 0.0)
    except (TypeError, ValueError):
        impact = 0.0
    action_type = str(item.get("action_type") or item.get("action") or "")
    novelty = 1.0 - _action_type_obviousness(action_type)
    return (priority, _tier_score_for_item(item), novelty, score, impact)


def _final_recommendation_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    primary_process = str(
        item.get("primary_process_id")
        or _recommendation_process_hint(item)
        or _item_process_norm(item)
        or ""
    ).strip().lower()
    plugin_id = str(item.get("plugin_id") or "").strip().lower()
    kind = str(item.get("kind") or "").strip().lower()

    if _ranking_version() == "v1":
        return (
            -float(_safe_num(item.get("modeled_delta_hours")) or 0.0),
            -float(_safe_num(item.get("relevance_score")) or 0.0),
            primary_process,
            plugin_id,
            kind,
        )

    weighted_rank = float(_safe_num(item.get("weighted_rank_score")) or 0.0)
    value_v2 = float(_safe_num(item.get("value_score_v2")) or 0.0)
    metric_confidence = float(
        _safe_num(item.get("metric_confidence"))
        or _safe_num(item.get("confidence_weight"))
        or 0.0
    )
    # Prefer single-process recommendations over grouped (lower = better).
    scope_rank = 0 if item.get("scope_type") == "single_process" else 1
    return (
        -weighted_rank,
        -value_v2,
        -metric_confidence,
        scope_rank,
        primary_process,
        plugin_id,
        kind,
    )


# ---------------------------------------------------------------------------
# Capacity scaling
# ---------------------------------------------------------------------------


def _capacity_scale_recommendation(
    kind: str | None, matched: list[dict[str, Any]], label: str, process_hint: str
) -> tuple[str | None, dict[str, Any]]:
    if not _include_capacity_recommendations():
        return None, {}
    if not kind or not matched:
        return None, {}
    suffix = f" (process {process_hint})" if process_hint else ""
    item = matched[0]
    meta: dict[str, Any] = {"action": None, "modeled_delta": None}
    if kind == "capacity_scale_model":
        base = item.get("eligible_wait_gt_hours_without_target")
        modeled = item.get("eligible_wait_gt_hours_modeled")
        scale = item.get("scale_factor")
        if isinstance(base, (int, float)) and isinstance(modeled, (int, float)):
            delta = float(base) - float(modeled)
            scale_text = f" scale_factor={scale:.3f}" if isinstance(scale, (int, float)) else ""
            meta = {"action": "add_one_server", "modeled_delta": delta}
            return (
                f"Add one server{suffix}: modeled >threshold eligible-wait drops "
                f"from {float(base):.3f}h to {float(modeled):.3f}h (Δ {delta:.3f}h){scale_text}."
            ), meta
    if kind == "capacity_scaling":
        base = item.get("baseline_wait_hours")
        modeled = item.get("modeled_wait_hours")
        reduction = item.get("reduction_hours")
        scale = item.get("scale_factor")
        if isinstance(base, (int, float)) and isinstance(modeled, (int, float)):
            reduction_val = reduction if isinstance(reduction, (int, float)) else float(base) - float(modeled)
            scale_text = f" scale_factor={scale:.3f}" if isinstance(scale, (int, float)) else ""
            meta = {"action": "add_one_server", "modeled_delta": float(reduction_val)}
            return (
                f"Add one server{suffix}: eligible-wait drops from "
                f"{float(base):.3f}h to {float(modeled):.3f}h (Δ {float(reduction_val):.3f}h){scale_text}."
            ), meta
    if kind == "close_cycle_capacity_impact":
        effect = item.get("effect")
        decision = str(item.get("decision") or "")
        if isinstance(effect, (int, float)) and decision == "detected":
            pct = abs(float(effect)) * 100.0
            meta = {"action": "add_one_server", "modeled_delta": None}
            return f"Add one server{suffix}: median close-cycle improves by ~{pct:.1f}% (measured).", meta
    return None, {}


# ---------------------------------------------------------------------------
# Deduplication & splitting
# ---------------------------------------------------------------------------


def _dedupe_recommendations(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, float], dict[str, Any]] = {}
    passthrough: list[dict[str, Any]] = []
    for item in items:
        action = item.get("action")
        delta = item.get("modeled_delta")
        if not action or delta is None:
            passthrough.append(item)
            continue
        try:
            delta_key = round(float(delta), 3)
        except (TypeError, ValueError):
            passthrough.append(item)
            continue
        key = (str(action), delta_key)
        if key not in merged:
            merged_item = dict(item)
            merged_item["merged_titles"] = [item.get("title") or ""]
            merged[key] = merged_item
            continue
        current = merged[key]
        current["merged_titles"].append(item.get("title") or "")
        current["observed_count"] = int(current.get("observed_count") or 0) + int(
            item.get("observed_count") or 0
        )
        if isinstance(current.get("evidence"), list) and isinstance(
            item.get("evidence"), list
        ):
            merged_evidence = current["evidence"] + item["evidence"]
            current["evidence"] = merged_evidence[:3]
    return list(merged.values()) + passthrough


def _recommendation_merge_key(item: dict[str, Any]) -> tuple[str, str]:
    action_type = str(item.get("action_type") or item.get("action") or "").strip().lower()
    target_ids = item.get("target_process_ids")
    if isinstance(target_ids, (list, tuple, set)):
        target_key = ",".join(sorted(str(t).strip().lower() for t in target_ids if t))
    else:
        proc = str(
            item.get("primary_process_id")
            or _recommendation_process_hint(item)
            or _item_process_norm(item)
            or ""
        ).strip().lower()
        target_key = proc
    scope_class = str(item.get("scope_class") or "").strip().lower()

    if not action_type:
        text = str(item.get("recommendation") or "").strip().lower()
        title = str(item.get("title") or "").strip().lower()
        if text.startswith("add one server") or "3rd server" in title or "third server" in title:
            action_type = "capacity_add_server"
        elif "close-cycle capacity" in title or "close-cycle capacity" in text:
            action_type = "close_cycle_capacity"
        elif text:
            action_type = text[:80]
        else:
            action_type = title[:80]

    merge_base = f"{action_type}:{target_key}:{scope_class}"
    delta = item.get("modeled_delta")
    delta_text = _format_issue_value(delta) if delta is not None else ""
    return merge_base, delta_text


def _dedupe_recommendations_text(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        key = _recommendation_merge_key(item)
        if key not in merged:
            merged_item = dict(item)
            merged_item["merged_titles"] = [item.get("title") or ""]
            merged[key] = merged_item
            continue
        current = merged[key]
        current["merged_titles"].append(item.get("title") or "")
    return list(merged.values())


def _split_recommendations(
    recommendations: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
    known_items: list[dict[str, Any]] = []
    discovery_items: list[dict[str, Any]] = []
    known_summary = ""
    discovery_summary = ""
    if not isinstance(recommendations, dict):
        return known_items, discovery_items, known_summary, discovery_summary
    if "known" in recommendations or "discovery" in recommendations:
        known_block = recommendations.get("known") or {}
        discovery_block = recommendations.get("discovery") or {}
        known_items = [
            item for item in (known_block.get("items") or []) if isinstance(item, dict)
        ]
        discovery_items = [
            item
            for item in (discovery_block.get("items") or [])
            if isinstance(item, dict)
        ]
        known_summary = known_block.get("summary") or ""
        discovery_summary = discovery_block.get("summary") or ""
        return known_items, discovery_items, known_summary, discovery_summary
    known_items = [
        item for item in (recommendations.get("items") or []) if isinstance(item, dict)
    ]
    known_summary = recommendations.get("summary") or ""
    return known_items, discovery_items, known_summary, discovery_summary


# ---------------------------------------------------------------------------
# Confidence & controllability weights
# ---------------------------------------------------------------------------


def _confidence_weight(item: dict[str, Any], issue: dict[str, Any]) -> float:
    for key in ("confidence_weight", "confidence"):
        value = issue.get(key) if isinstance(issue, dict) else None
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
    value = item.get("confidence")
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    p_value = item.get("p_value")
    if isinstance(p_value, (int, float)):
        return confidence_from_p(float(p_value))
    return 0.5


def _controllability_weight(kind: str | None, issue: dict[str, Any]) -> float:
    override = issue.get("controllability_weight") if isinstance(issue, dict) else None
    if isinstance(override, (int, float)):
        return max(0.0, min(1.0, float(override)))
    mapping = {
        "capacity_scale_model": 0.9,
        "capacity_scaling": 0.9,
        "close_cycle_capacity_model": 0.8,
        "close_cycle_capacity_impact": 0.8,
        "eligible_wait_process_stats": 0.7,
        "eligible_wait_impact": 0.7,
        "close_cycle_revenue_compression": 0.6,
        "close_cycle_duration_shift": 0.7,
        "upload_bkrvnu_linkage": 0.6,
        "process_counterfactual": 0.6,
        "sequence_bottleneck": 0.55,
        "user_host_savings": 0.5,
    }
    return mapping.get(kind or "", 0.5)
