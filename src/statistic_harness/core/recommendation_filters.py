from __future__ import annotations

from typing import Any


FLOW_REWIRE_ACTION_TYPES = {
    "unblock_dependency_chain",
    "orchestrate_chain",
    "reduce_transition_gap",
    "route_process",
}

NON_SPECIFIC_PROCESS_TOKENS = {
    "",
    "(multiple)",
    "multiple",
    "all",
    "any",
    "global",
}


def normalize_process_token(raw: Any) -> str:
    token = str(raw or "").strip().lower()
    if token.startswith("proc:"):
        token = token[5:].strip()
    return token


def is_specific_process_target(raw: Any) -> bool:
    return normalize_process_token(raw) not in NON_SPECIFIC_PROCESS_TOKENS


def is_flow_rewire_action(action_type: Any) -> bool:
    token = str(action_type or "").strip().lower()
    return token in FLOW_REWIRE_ACTION_TYPES


def process_is_adjustable(process_id: Any, non_adjustable: set[str]) -> bool:
    token = normalize_process_token(process_id)
    if token in NON_SPECIFIC_PROCESS_TOKENS:
        return False
    return token not in {str(v).strip().lower() for v in (non_adjustable or set())}

