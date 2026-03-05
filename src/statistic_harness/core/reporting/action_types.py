from __future__ import annotations

import re
from typing import Any


def _infer_action_type_from_text(text: str) -> str:
    token = str(text or "").strip().lower()
    if not token:
        return ""
    if "add" in token and ("server" in token or "capacity" in token):
        return "add_server"
    if "batch" in token and "group" in token:
        return "batch_group_candidate"
    if "batch" in token:
        return "batch_input"
    if "cache" in token:
        return "batch_or_cache"
    if "resched" in token or "schedule" in token:
        return "reschedule"
    if "route" in token:
        return "route_process"
    if "throttle" in token or "dedupe" in token or "de-duplicate" in token:
        return "throttle_or_dedupe"
    return ""


def _normalize_action_type(raw_action: str, text_hint: str = "") -> str:
    action = str(raw_action or "").strip().lower()
    if not action:
        inferred = _infer_action_type_from_text(text_hint)
        return inferred
    alias_map = {
        "batch_cluster_candidate": "batch_group_candidate",
        "preset_job_candidate": "batch_input_refactor",
        "param_rule_simplification": "batch_input_refactor",
        "action_plan_combo": "batch_group_candidate",
        "simulate_plan": "batch_or_cache",
        "reduce_close_cycle_slowdown": "reschedule",
        "isolate_process": "tune_schedule",
        "orchestrate_macro": "schedule_shift_target",
    }
    return str(alias_map.get(action, action))


def _flow_rewire_action_types() -> set[str]:
    # Processes in parent/child chains can still take process-local changes.
    # Only drop actions that require flow rewiring across process boundaries.
    return {
        "unblock_dependency_chain",
        "orchestrate_chain",
        "reduce_transition_gap",
        "route_process",
    }


def _action_type_tier(action_type: str) -> int:
    """Recommendation tiering to keep outputs non-generic and structurally actionable.

    Tier 1: structural/interface changes (batch/multi-input, dedupe/caching, macro consolidation)
    Tier 2: targeted operational adjustments (schedule/route) with clear evidence
    Tier 3: generic tuning ("make it faster", thresholds) - usually suppressed/capped
    """

    at = (action_type or "").strip().lower()
    if not at:
        return 2
    tier1 = {
        "batch_input",
        "batch_or_cache",
        "batch_input_refactor",
        "dedupe_or_cache",
        "unblock_dependency_chain",
        "reduce_transition_gap",
        "orchestrate_chain",
        "orchestrate_macro",
        "decouple_boundary",
        "shared_cache_endpoint",
        "batch_group_candidate",
        "cluster_with_constraints",
        "reduce_spillover_past_eom",
    }
    tier2 = {
        "schedule_shift_target",
        "reschedule",
        "route_process",
        "rebalance_assignment",
    }
    tier3 = {
        "reduce_process_wait",
        "review",
        "tune_threshold",
    }
    if at in tier1:
        return 1
    if at in tier2:
        return 2
    if at in tier3:
        return 3
    # Default to Tier 2 if unknown but actionable.
    return 2


def _tier_score_for_item(item: dict[str, Any]) -> int:
    # Higher score sorts first (used under reverse=True).
    at = str(item.get("action_type") or item.get("action") or "").strip()
    tier = _action_type_tier(at)
    if tier == 1:
        return 3
    if tier == 2:
        return 2
    if tier == 3:
        return 1
    return 1


def _action_type_obviousness(action_type: str) -> float:
    at = (action_type or "").strip().lower()
    if at in {
        "batch_group_candidate",
        "unblock_dependency_chain",
        "reduce_transition_gap",
        "cluster_with_constraints",
        "distribution_shift_target",
        "burst_trigger",
    }:
        return 0.20
    if at in {
        "batch_input",
        "batch_input_refactor",
        "batch_or_cache",
        "dedupe_or_cache",
        "orchestrate_chain",
        "orchestrate_macro",
        "decouple_boundary",
        "shared_cache_endpoint",
    }:
        return 0.30
    if at in {"route_process", "rebalance_assignment", "rebalance_group"}:
        return 0.60
    if at in {"schedule_shift_target", "reschedule"}:
        return 0.70
    if at in {"reduce_process_wait", "review", "tune_threshold"}:
        return 0.95
    if at in {"reduce_spillover_past_eom"}:
        return 0.70
    if not at:
        return 0.85
    return 0.65


def _obviousness_rank(score: float) -> str:
    if score <= 0.35:
        return "needle"
    if score <= 0.70:
        return "targeted"
    return "obvious"
