from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Module-level constants (env-var names, paths, caches)
# ---------------------------------------------------------------------------

_RANKING_WEIGHTS_PATH = Path(__file__).resolve().parents[4] / "config" / "recommendation_weights.yaml"
_RANKING_WEIGHTS_CACHE: dict[str, float] | None = None

_INCLUDE_KNOWN_RECOMMENDATIONS_ENV = "STAT_HARNESS_INCLUDE_KNOWN_RECOMMENDATIONS"
_ALLOW_KNOWN_ISSUE_SYNTHETIC_MATCHES_ENV = "STAT_HARNESS_ALLOW_KNOWN_ISSUE_SYNTHETIC_MATCHES"
_SUPPRESS_ACTION_TYPES_ENV = "STAT_HARNESS_SUPPRESS_ACTION_TYPES"
_MAX_PER_ACTION_TYPE_ENV = "STAT_HARNESS_MAX_PER_ACTION_TYPE"
_ALLOW_ACTION_TYPES_ENV = "STAT_HARNESS_ALLOW_ACTION_TYPES"
_ALLOW_PROCESS_PATTERNS_ENV = "STAT_HARNESS_RECOMMENDATION_ALLOW_PROCESSES"
_MIN_RELEVANCE_SCORE_ENV = "STAT_HARNESS_RECOMMENDATION_MIN_RELEVANCE"
_DISCOVERY_TOP_N_ENV = "STAT_HARNESS_DISCOVERY_TOP_N"
_MAX_OBVIOUSNESS_ENV = "STAT_HARNESS_MAX_OBVIOUSNESS"
_RECENCY_UNKNOWN_WEIGHT_ENV = "STAT_HARNESS_RECENCY_UNKNOWN_WEIGHT"
_RECENCY_DECAY_PER_MONTH_ENV = "STAT_HARNESS_RECENCY_DECAY_PER_MONTH"
_RECENCY_MIN_WEIGHT_ENV = "STAT_HARNESS_RECENCY_MIN_WEIGHT"
_REQUIRE_MODELED_HOURS_ENV = "STAT_HARNESS_REQUIRE_MODELED_HOURS"
_REQUIRE_DIRECT_PROCESS_ACTION_ENV = "STAT_HARNESS_REQUIRE_DIRECT_PROCESS_ACTION"
_PRE_REPORT_FILTER_MODE_ENV = "STAT_HARNESS_PRE_REPORT_FILTER_MODE"
_DIRECT_PROCESS_ACTION_TYPES_ENV = "STAT_HARNESS_DIRECT_PROCESS_ACTION_TYPES"
_CHAIN_BOUND_RATIO_MIN_ENV = "STAT_HARNESS_CHAIN_BOUND_RATIO_MIN"
_RANKING_VERSION_ENV = "STAT_HARNESS_RANKING_VERSION"
_CLOSE_CYCLES_PER_YEAR_ENV = "STAT_HARNESS_CLOSE_CYCLES_PER_YEAR"
_REPORT_MD_MAX_FINDINGS_PER_PLUGIN_ENV = "STAT_HARNESS_REPORT_MD_MAX_FINDINGS_PER_PLUGIN"
_REPORT_MD_MAX_STRING_LEN_ENV = "STAT_HARNESS_REPORT_MD_MAX_STRING_LEN"
_REPORT_MD_MAX_EVIDENCE_IDS_ENV = "STAT_HARNESS_REPORT_MD_MAX_EVIDENCE_IDS"
_PLUGIN_CLASS_TAXONOMY_PATH = Path(__file__).resolve().parents[4] / "docs" / "plugin_class_taxonomy.yaml"
_PLUGIN_CLASS_TAXONOMY_CACHE: dict[str, Any] | None = None
_REPORTING_CONFIG_PATH = Path(__file__).resolve().parents[4] / "config" / "reporting.yaml"
_REPORTING_CONFIG_CACHE: dict[str, Any] | None = None

# ---------------------------------------------------------------------------
# Config reader functions
# ---------------------------------------------------------------------------


def _include_known_recommendations() -> bool:
    return os.environ.get(_INCLUDE_KNOWN_RECOMMENDATIONS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _allow_known_issue_synthetic_matches() -> bool:
    return os.environ.get(_ALLOW_KNOWN_ISSUE_SYNTHETIC_MATCHES_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _require_modeled_hours() -> bool:
    raw = os.environ.get(_REQUIRE_MODELED_HOURS_ENV, "").strip().lower()
    if not raw:
        return not _pre_report_passthrough_enabled()
    return raw in {"1", "true", "yes", "on"}


def _require_direct_process_action() -> bool:
    raw = os.environ.get(_REQUIRE_DIRECT_PROCESS_ACTION_ENV, "").strip().lower()
    if not raw:
        return not _pre_report_passthrough_enabled()
    return raw in {"1", "true", "yes", "on"}


def _pre_report_filter_mode() -> str:
    raw = os.environ.get(_PRE_REPORT_FILTER_MODE_ENV, "").strip().lower()
    if raw in {"strict", "passthrough"}:
        return raw
    return "passthrough"


def _pre_report_passthrough_enabled() -> bool:
    return _pre_report_filter_mode() != "strict"


def _direct_process_action_types() -> set[str]:
    raw = os.environ.get(_DIRECT_PROCESS_ACTION_TYPES_ENV, "").strip()
    if raw:
        out: set[str] = set()
        for token in re.split(r"[;,\s]+", raw):
            token = token.strip().lower()
            if token:
                out.add(token)
        if out:
            return out
    return {
        "add_server",
        "batch_input",
        "batch_group_candidate",
        "batch_or_cache",
        "batch_input_refactor",
        "throttle_or_dedupe",
        "dedupe_or_cache",
        "orchestrate_macro",
        "ideaspace_action",
        "route_process",
        "reschedule",
        "tune_schedule",
        "schedule_shift_target",
        "reduce_spillover_past_eom",
        "add_upload_linkage",
        "reduce_close_cycle_slowdown",
        "isolate_process",
        "cap_concurrency",
    }


def _chain_bound_ratio_min() -> float:
    raw = os.environ.get(_CHAIN_BOUND_RATIO_MIN_ENV, "").strip()
    if not raw:
        return 0.5
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.5
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _ranking_version() -> str:
    raw = os.environ.get(_RANKING_VERSION_ENV, "").strip().lower()
    if raw in {"v1", "legacy"}:
        return "v1"
    return "v2"


def _close_cycles_per_year() -> float:
    raw = os.environ.get(_CLOSE_CYCLES_PER_YEAR_ENV, "").strip()
    if not raw:
        return 12.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 12.0
    if value <= 0.0:
        return 12.0
    return float(value)


def _read_positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return int(value) if value > 0 else int(default)


def _env_float(name: str, default: float, *, min_value: float | None = None, max_value: float | None = None) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    if min_value is not None and value < float(min_value):
        value = float(min_value)
    if max_value is not None and value > float(max_value):
        value = float(max_value)
    return float(value)


def _include_capacity_recommendations() -> bool:
    # Capacity suggestions like "add one server" / "QPEC+1" are usually generic. Keep them opt-in.
    return os.environ.get("STAT_HARNESS_INCLUDE_CAPACITY_RECOMMENDATIONS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _suppressed_action_types() -> set[str]:
    if _pre_report_passthrough_enabled():
        return set()
    # Strict-mode default: hide only the truly generic tuning knobs; keep structural levers visible.
    defaults = {"tune_threshold"}
    raw = os.environ.get(_SUPPRESS_ACTION_TYPES_ENV, "").strip()
    if raw == "":
        return defaults
    out: set[str] = set()
    for token in re.split(r"[;,\\s]+", raw):
        token = token.strip()
        if token:
            out.add(token)
    return out


def _max_per_action_type() -> dict[str, int]:
    if _pre_report_passthrough_enabled():
        return {}
    # Breadth-first defaults: prevent one action type from drowning out the rest.
    defaults: dict[str, int] = {
        "batch_input": 8,
        "batch_or_cache": 6,
        "batch_input_refactor": 6,
        "dedupe_or_cache": 4,
        "unblock_dependency_chain": 6,
        "reduce_transition_gap": 6,
        "orchestrate_chain": 5,
        "orchestrate_macro": 5,
        "decouple_boundary": 4,
        "shared_cache_endpoint": 4,
        "batch_group_candidate": 4,
        "cluster_with_constraints": 3,
        "distribution_shift_target": 4,
        "burst_trigger": 4,
        "schedule_shift_target": 4,
        "reschedule": 3,
        "route_process": 3,
        "reduce_process_wait": 2,
        "add_instrumentation": 1,
        "review": 2,
        "tune_threshold": 1,
    }
    raw = os.environ.get(_MAX_PER_ACTION_TYPE_ENV, "").strip()
    if not raw:
        return defaults
    out = dict(defaults)
    for token in re.split(r"[;,\\s]+", raw):
        token = token.strip()
        if not token or "=" not in token:
            continue
        k, v = token.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        try:
            out[k] = int(v)
        except (TypeError, ValueError):
            continue
    return out


def _allow_action_types() -> set[str]:
    raw = os.environ.get(_ALLOW_ACTION_TYPES_ENV, "").strip()
    if not raw:
        return set()
    out: set[str] = set()
    for token in re.split(r"[;,\s]+", raw):
        token = token.strip()
        if token:
            out.add(token)
    return out


def _allow_process_patterns() -> list[str]:
    raw = os.environ.get(_ALLOW_PROCESS_PATTERNS_ENV, "").strip()
    if not raw:
        return []
    out: list[str] = []
    for token in re.split(r"[;,\s]+", raw):
        token = token.strip()
        if token:
            out.append(token)
    return out


def _min_relevance_score() -> float:
    raw = os.environ.get(_MIN_RELEVANCE_SCORE_ENV, "").strip()
    if not raw:
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return value if value > 0 else 0.0


def _discovery_top_n() -> int | None:
    raw = os.environ.get(_DISCOVERY_TOP_N_ENV, "").strip()
    if raw:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 0
        return value if value > 0 else None
    cfg = _reporting_config()
    cfg_top_n = cfg.get("recommendation_top_n") if isinstance(cfg, dict) else None
    if isinstance(cfg_top_n, (int, float)) and int(cfg_top_n) > 0:
        return int(cfg_top_n)
    return None


def _reporting_config() -> dict[str, Any]:
    global _REPORTING_CONFIG_CACHE
    if _REPORTING_CONFIG_CACHE is not None:
        return dict(_REPORTING_CONFIG_CACHE)
    if not _REPORTING_CONFIG_PATH.exists():
        _REPORTING_CONFIG_CACHE = {}
        return {}
    try:
        payload = yaml.safe_load(_REPORTING_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    _REPORTING_CONFIG_CACHE = payload if isinstance(payload, dict) else {}
    return dict(_REPORTING_CONFIG_CACHE)


def _max_obviousness() -> float:
    if _pre_report_passthrough_enabled():
        return 1.0
    # Lower = more "needle in haystack". Strict mode defaults to filtering out generic obvious actions.
    raw = os.environ.get(_MAX_OBVIOUSNESS_ENV, "").strip()
    if not raw:
        return 0.74
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.74
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _recommendation_controls(report: dict[str, Any]) -> dict[str, Any]:
    known = report.get("known_issues") if isinstance(report.get("known_issues"), dict) else None
    if not isinstance(known, dict):
        return {}
    controls = known.get("recommendation_controls")
    return controls if isinstance(controls, dict) else {}
