from __future__ import annotations

import csv
from collections import Counter, deque
import json
import re
import os
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from jsonschema import validate
import yaml

from .four_pillars import build_four_pillars_scorecard
from .actionability_explanations import (
    NON_ADJUSTABLE_PROCESSES,
    derive_reason_code,
    plain_english_explanation,
    recommended_next_step,
)
from .known_issues_mode import known_issues_enabled, known_issues_mode_label
from .stat_controls import confidence_from_p
from .process_matcher import (
    compile_patterns,
    default_exclude_process_patterns,
    parse_exclude_patterns_env,
)
from .storage import Storage
from .utils import atomic_write_text, json_dumps, now_iso, read_json, write_json
from .accounting_windows import (
    AccountingWindow,
    infer_accounting_windows_from_timestamps,
    load_accounting_windows_from_run,
    window_ranges,
)

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
_DIRECT_PROCESS_ACTION_TYPES_ENV = "STAT_HARNESS_DIRECT_PROCESS_ACTION_TYPES"
_CHAIN_BOUND_RATIO_MIN_ENV = "STAT_HARNESS_CHAIN_BOUND_RATIO_MIN"
_RANKING_VERSION_ENV = "STAT_HARNESS_RANKING_VERSION"
_CLOSE_CYCLES_PER_YEAR_ENV = "STAT_HARNESS_CLOSE_CYCLES_PER_YEAR"
_REPORT_MD_MAX_FINDINGS_PER_PLUGIN_ENV = "STAT_HARNESS_REPORT_MD_MAX_FINDINGS_PER_PLUGIN"
_REPORT_MD_MAX_STRING_LEN_ENV = "STAT_HARNESS_REPORT_MD_MAX_STRING_LEN"
_REPORT_MD_MAX_EVIDENCE_IDS_ENV = "STAT_HARNESS_REPORT_MD_MAX_EVIDENCE_IDS"
_PLUGIN_CLASS_TAXONOMY_PATH = Path(__file__).resolve().parents[3] / "docs" / "plugin_class_taxonomy.yaml"
_PLUGIN_CLASS_TAXONOMY_CACHE: dict[str, Any] | None = None


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
        return True
    return raw in {"1", "true", "yes", "on"}


def _require_direct_process_action() -> bool:
    raw = os.environ.get(_REQUIRE_DIRECT_PROCESS_ACTION_ENV, "").strip().lower()
    if not raw:
        return True
    return raw in {"1", "true", "yes", "on"}


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
        "route_process",
        "reschedule",
        "tune_schedule",
    }


def _normalize_process_hint(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    token = raw.strip().lower()
    if token.startswith("proc:"):
        token = token[5:].strip()
    if token in {"", "(multiple)", "multiple", "all", "any", "global"}:
        return ""
    return token


def _finding_recommendation_text(finding: dict[str, Any]) -> str:
    text = str(finding.get("recommendation") or "").strip()
    if text:
        return text
    recs = finding.get("recommendations")
    if isinstance(recs, list):
        for value in recs:
            token = str(value or "").strip()
            if token:
                return token
    return ""


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


def _target_process_ids_for_finding(finding: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key in ("process_norm", "process", "process_id"):
        token = _normalize_process_hint(finding.get(key))
        if token:
            out.append(token)
    evidence = finding.get("evidence") if isinstance(finding.get("evidence"), dict) else {}
    for bucket in (finding, evidence):
        raw = bucket.get("target_process_ids") if isinstance(bucket, dict) else None
        if isinstance(raw, list):
            for value in raw:
                token = _normalize_process_hint(value)
                if token:
                    out.append(token)
    selected = evidence.get("selected") if isinstance(evidence.get("selected"), list) else []
    for row in selected:
        if not isinstance(row, dict):
            continue
        for key in ("process_norm", "process", "process_id"):
            token = _normalize_process_hint(row.get(key))
            if token:
                out.append(token)
    for key in ("processes", "process_ids"):
        raw_list = evidence.get(key) if isinstance(evidence.get(key), list) else []
        for value in raw_list:
            token = _normalize_process_hint(value)
            if token:
                out.append(token)
    deduped: list[str] = []
    seen: set[str] = set()
    for token in out:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


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


def _trim_long_text(value: Any, *, max_chars: int) -> Any:
    if isinstance(value, str) and len(value) > max_chars:
        suffix = f"... [truncated {len(value) - max_chars} chars]"
        return value[:max_chars] + suffix
    return value


def _trim_finding_for_markdown(
    finding: Any,
    *,
    max_chars: int,
    max_evidence_ids: int,
) -> Any:
    if not isinstance(finding, dict):
        return _trim_long_text(finding, max_chars=max_chars)
    out: dict[str, Any] = {}
    for key, value in finding.items():
        if isinstance(value, str):
            out[key] = _trim_long_text(value, max_chars=max_chars)
            continue
        if key == "evidence" and isinstance(value, dict):
            ev = dict(value)
            for id_key in ("row_ids", "column_ids"):
                ids = ev.get(id_key)
                if isinstance(ids, list) and len(ids) > max_evidence_ids:
                    omitted = len(ids) - max_evidence_ids
                    ev[id_key] = list(ids[:max_evidence_ids]) + [f"... ({omitted} more)"]
            out[key] = ev
            continue
        out[key] = value
    return out


def _trim_plugin_dump_for_markdown(plugin_payload: Any) -> Any:
    if not isinstance(plugin_payload, dict):
        return plugin_payload
    max_findings = _read_positive_int_env(_REPORT_MD_MAX_FINDINGS_PER_PLUGIN_ENV, 120)
    max_chars = _read_positive_int_env(_REPORT_MD_MAX_STRING_LEN_ENV, 500)
    max_evidence_ids = _read_positive_int_env(_REPORT_MD_MAX_EVIDENCE_IDS_ENV, 20)

    out = dict(plugin_payload)
    findings = plugin_payload.get("findings")
    if isinstance(findings, list):
        trimmed_findings = [
            _trim_finding_for_markdown(
                item,
                max_chars=max_chars,
                max_evidence_ids=max_evidence_ids,
            )
            for item in findings[:max_findings]
        ]
        if len(findings) > max_findings:
            trimmed_findings.append(
                {
                    "kind": "report_dump_truncated",
                    "reason": f"trimmed to first {max_findings} findings for report.md",
                    "omitted_findings": len(findings) - max_findings,
                }
            )
        out["findings"] = trimmed_findings
    summary = out.get("summary")
    if isinstance(summary, str):
        out["summary"] = _trim_long_text(summary, max_chars=max_chars)
    return out


def _infer_ideaspace_roles(columns_index: list[dict[str, Any]]) -> dict[str, Any]:
    """Infer dataset "roles" from column metadata for ideaspace reporting."""

    names = [str(col.get("name") or "") for col in columns_index if isinstance(col, dict)]
    lowered = [n.lower() for n in names]
    roles = [str(col.get("role") or "").lower() for col in columns_index if isinstance(col, dict)]

    def pick(tokens: tuple[str, ...]) -> str | None:
        for name in names:
            lname = name.lower()
            if any(tok in lname for tok in tokens):
                return name
        return None

    time_col = pick(("time", "timestamp", "date", "created", "updated", "start", "end"))
    process_col = pick(("process", "activity", "task", "action", "job", "step", "workflow"))
    host_col = pick(("host", "server", "node", "instance", "machine"))
    user_col = pick(("user", "owner", "operator", "agent"))
    params = []
    for name in names:
        lname = name.lower()
        if any(tok in lname for tok in ("param", "type", "code", "variant", "reason")):
            params.append(name)
            if len(params) >= 5:
                break
    case_id_col = pick(("case", "trace", "span", "correlation", "request_id", "session"))
    if not case_id_col:
        # If a column is explicitly tagged as ID in a template, surface it.
        for idx, role in enumerate(roles):
            if role and "id" in role and idx < len(names):
                case_id_col = names[idx]
                break

    has_coords = any(any(tok in n for tok in ("lat", "lon", "coord", "longitude", "latitude")) for n in lowered)
    has_text = any(any(tok in n for tok in ("message", "error", "exception", "trace", "stack", "log")) for n in lowered)
    return {
        "time_column": time_col,
        "process_column": process_col,
        "host_column": host_col,
        "user_column": user_col,
        "params_columns": params,
        "case_id_column": case_id_col,
        "has_coords": has_coords,
        "has_text": has_text,
    }


def _ideaspace_families_summary(plugins: dict[str, Any]) -> list[dict[str, Any]]:
    """Summarize applicability (ok/na) of idea families A-F."""

    families: dict[str, list[str]] = {
        "A_tda": [
            "analysis_tda_persistent_homology",
            "analysis_tda_persistence_landscapes",
            "analysis_tda_mapper_graph",
            "analysis_tda_betti_curve_changepoint",
        ],
        "B_topographic": [
            "analysis_topographic_similarity_angle_projection",
            "analysis_topographic_angle_dynamics",
            "analysis_topographic_tanova_permutation",
            "analysis_map_permutation_test_karniski",
        ],
        "C_surface": [
            "analysis_surface_multiscale_wavelet_curvature",
            "analysis_surface_fractal_dimension_variogram",
            "analysis_surface_rugosity_index",
            "analysis_surface_terrain_position_index",
            "analysis_surface_fabric_sso_eigen",
            "analysis_surface_hydrology_flow_watershed",
            "analysis_surface_roughness_metrics",
            "analysis_monte_carlo_surface_uncertainty",
        ],
        "D_classic_auto": [
            "analysis_ttests_auto",
            "analysis_chi_square_association",
            "analysis_anova_auto",
            "analysis_regression_auto",
            "analysis_time_series_analysis_auto",
            "analysis_cluster_analysis_auto",
            "analysis_pca_auto",
        ],
        "E_uncertainty": [
            "analysis_bayesian_point_displacement",
        ],
        "F_ops_levers": [
            "analysis_actionable_ops_levers_v1",
        ],
    }

    out: list[dict[str, Any]] = []
    for family, pids in families.items():
        present = []
        for pid in pids:
            plugin = plugins.get(pid) if isinstance(plugins, dict) else None
            if not isinstance(plugin, dict):
                continue
            present.append(
                {
                    "plugin_id": pid,
                    "status": plugin.get("status"),
                    "summary": plugin.get("summary"),
                }
            )
        if not present:
            continue
        statuses = [str(p.get("status") or "") for p in present]
        applicable = any(s == "ok" for s in statuses)
        all_na = all(s in {"na", "not_applicable"} for s in statuses)
        reason = None
        if all_na:
            reason = next((p.get("summary") for p in present if isinstance(p.get("summary"), str)), None)
        out.append(
            {
                "family": family,
                "plugins": present,
                "applicable": bool(applicable),
                "all_not_applicable": bool(all_na),
                "reason": reason,
            }
        )
    return out


def _matches_expected(
    item: dict[str, Any],
    where: dict[str, Any] | None,
    contains: dict[str, Any] | None,
) -> bool:
    if where:
        for key, expected in where.items():
            actual = item.get(key)
            if actual != expected:
                return False
    if contains:
        for key, expected in contains.items():
            actual = item.get(key)
            if isinstance(actual, str):
                if str(expected) not in actual:
                    return False
            elif isinstance(actual, (list, tuple, set)):
                if isinstance(expected, (list, tuple, set)):
                    if not set(expected).issubset(set(actual)):
                        return False
                else:
                    if expected not in actual:
                        return False
            else:
                return False
    return True


def _collect_findings_for_plugin(
    report: dict[str, Any], plugin_id: str | None, kind: str | None = None
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    plugins = report.get("plugins", {}) or {}
    for pid, plugin in plugins.items():
        if plugin_id and pid != plugin_id:
            continue
        for item in plugin.get("findings", []) or []:
            if kind and item.get("kind") != kind:
                continue
            if isinstance(item, dict):
                findings.append(item)
    return findings


def _process_hint(where: dict[str, Any] | None) -> str:
    if not where:
        return ""
    for key in ("process", "process_norm", "process_name", "activity"):
        value = where.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _known_recommendation_match(
    report: dict[str, Any], plugin_id: str | None, kind: str | None, process_hint: str
) -> dict[str, Any] | None:
    recommendations = report.get("recommendations")
    if not isinstance(recommendations, dict):
        return None
    known_block = recommendations.get("known")
    if not isinstance(known_block, dict):
        return None
    items = known_block.get("items")
    if not isinstance(items, list):
        return None
    proc = str(process_hint or "").strip().lower()
    for item in items:
        if not isinstance(item, dict):
            continue
        if plugin_id and str(item.get("plugin_id") or "").strip() != str(plugin_id).strip():
            continue
        if kind and str(item.get("kind") or "").strip() != str(kind).strip():
            continue
        if proc and _item_process_norm(item) != proc:
            continue
        return item
    return None


def _synthetic_close_cycle_contention_finding(
    report: dict[str, Any], storage: Storage | None, process_hint: str
) -> dict[str, Any] | None:
    if storage is None:
        return None
    proc = str(process_hint or "").strip().lower()
    if not proc:
        return None
    model = _process_removal_model(report, storage, proc)
    if not isinstance(model, dict):
        return None
    close_hours = model.get("close_delta_hours")
    general_hours = model.get("general_delta_hours")
    close_pct = model.get("close_modeled_percent")
    general_pct = model.get("general_modeled_percent")
    if not any(isinstance(v, (int, float)) and float(v) > 0.0 for v in (close_hours, general_hours, close_pct, general_pct)):
        return None
    best_pct = 0.0
    for value in (close_pct, general_pct):
        if isinstance(value, (int, float)):
            best_pct = max(best_pct, float(value) / 100.0)
    modeled_hours = 0.0
    for value in (close_hours, general_hours):
        if isinstance(value, (int, float)):
            modeled_hours = max(modeled_hours, float(value))
    return {
        "kind": "close_cycle_contention",
        "measurement_type": "modeled",
        "process": proc,
        "process_norm": proc,
        "estimated_improvement_pct": best_pct,
        "modeled_reduction_pct": best_pct,
        "modeled_reduction_hours": modeled_hours,
        "modeled_assumption": "known_issue_process_removal_model",
        "modeled_close_percent": float(close_pct) if isinstance(close_pct, (int, float)) else None,
        "modeled_general_percent": float(general_pct) if isinstance(general_pct, (int, float)) else None,
    }


def _recommendation_text(status: str, label: str, process_hint: str) -> str:
    suffix = f" (process {process_hint})" if process_hint else ""
    if status == "confirmed":
        return f"Act on {label}{suffix}."
    if status == "over_limit":
        return f"Investigate excess occurrences of {label}{suffix}."
    if status in {"missing", "below_min"}:
        return f"Missing evidence for {label}{suffix}; check inputs and re-run."
    return f"Review {label}{suffix}."


def _include_capacity_recommendations() -> bool:
    # Capacity suggestions like "add one server" / "QPEC+1" are usually generic. Keep them opt-in.
    return os.environ.get("STAT_HARNESS_INCLUDE_CAPACITY_RECOMMENDATIONS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _suppressed_action_types() -> set[str]:
    # Default: hide only the truly generic tuning knobs; keep structural levers visible.
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
    if not raw:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _max_obviousness() -> float:
    # Lower = more "needle in haystack". Defaults to filtering out generic obvious actions.
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


def _recommendation_controls(report: dict[str, Any]) -> dict[str, Any]:
    known = report.get("known_issues") if isinstance(report.get("known_issues"), dict) else None
    if not isinstance(known, dict):
        return {}
    controls = known.get("recommendation_controls")
    return controls if isinstance(controls, dict) else {}


def _recommendation_process_hint(item: dict[str, Any]) -> str:
    where = item.get("where") if isinstance(item.get("where"), dict) else None
    contains = item.get("contains") if isinstance(item.get("contains"), dict) else None
    hint = _process_hint(where) or _process_hint(contains)
    if isinstance(hint, str) and hint.strip():
        return hint.strip()
    evidence = item.get("evidence")
    if isinstance(evidence, list):
        for row in evidence:
            if not isinstance(row, dict):
                continue
            for key in ("process", "process_norm", "process_id"):
                val = row.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
    return ""


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

    value_score = float(_safe_num(item.get("value_score_v2")) or _safe_num(item.get("client_value_score")) or 0.0)
    close_dynamic_delta = float(_safe_num(item.get("delta_hours_close_dynamic")) or _safe_num(item.get("modeled_delta_hours")) or 0.0)
    close_dynamic_eff = float(_safe_num(item.get("efficiency_gain_close_dynamic")) or 0.0)
    acct_delta = float(_safe_num(item.get("delta_hours_accounting_month")) or _safe_num(item.get("modeled_delta_hours")) or 0.0)
    manual_pct = float(_safe_num(item.get("modeled_manual_run_reduction_pct")) or 0.0)
    metric_confidence = float(
        _safe_num(item.get("metric_confidence"))
        or _safe_num(item.get("confidence_weight"))
        or 0.0
    )
    scope_type = str(item.get("scope_type") or "").strip().lower()
    scope_rank = 0 if scope_type == "single_process" else 1
    return (
        -close_dynamic_delta,
        -close_dynamic_eff,
        -acct_delta,
        -value_score,
        -metric_confidence,
        scope_rank,
        -manual_pct,
        primary_process,
        plugin_id,
        kind,
    )


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


def _known_issue_processes(known: dict[str, Any] | None) -> set[str]:
    processes: set[str] = set()
    if not isinstance(known, dict):
        return processes
    expected = known.get("expected_findings") or []
    if not isinstance(expected, list):
        return processes
    keys = {"process", "process_norm", "process_name", "process_id", "activity", "process_matches"}
    for issue in expected:
        if not isinstance(issue, dict):
            continue
        for bucket in (issue.get("where"), issue.get("contains")):
            if not isinstance(bucket, dict):
                continue
            for key in keys:
                value = bucket.get(key)
                if isinstance(value, str) and value.strip():
                    processes.add(value.strip().lower())
                elif isinstance(value, (list, tuple, set)):
                    for entry in value:
                        if isinstance(entry, str) and entry.strip():
                            processes.add(entry.strip().lower())
    return processes


def _explicit_excluded_processes(report: dict[str, Any]) -> set[str]:
    excluded: set[str] = set()
    for entry in parse_exclude_patterns_env():
        if entry.strip():
            excluded.add(entry.strip().lower())
    known = report.get("known_issues") if isinstance(report.get("known_issues"), dict) else None
    if isinstance(known, dict):
        for key in ("exclude_processes", "excluded_processes"):
            values = known.get(key)
            if isinstance(values, (list, tuple, set)):
                for entry in values:
                    if isinstance(entry, str) and entry.strip():
                        excluded.add(entry.strip().lower())
        exclusions = known.get("recommendation_exclusions")
        if isinstance(exclusions, dict):
            values = exclusions.get("processes")
            if isinstance(values, (list, tuple, set)):
                for entry in values:
                    if isinstance(entry, str) and entry.strip():
                        excluded.add(entry.strip().lower())
    # Defaults: keep obvious "already-accounted-for" families out of recommendation budget
    # unless the operator explicitly provides their own list.
    if not excluded:
        excluded.update([p.strip().lower() for p in default_exclude_process_patterns() if p.strip()])
    return excluded


def _recommendation_has_excluded_process(
    item: dict[str, Any], excluded_match
) -> bool:
    if excluded_match is None:
        return False
    for bucket in (item.get("where"), item.get("contains")):
        if not isinstance(bucket, dict):
            continue
        for key in ("process", "process_norm", "process_name", "process_id", "activity"):
            value = bucket.get(key)
            if isinstance(value, str) and value.strip():
                if excluded_match(value.strip()):
                    return True
            elif isinstance(value, (list, tuple, set)):
                for entry in value:
                    if isinstance(entry, str) and entry.strip():
                        if excluded_match(entry.strip()):
                            return True
    return False


def _filter_recommendations_by_process(
    items: list[dict[str, Any]], excluded_patterns: set[str]
) -> list[dict[str, Any]]:
    if not excluded_patterns:
        return items
    excluded_match = compile_patterns(sorted(excluded_patterns))
    return [item for item in items if not _recommendation_has_excluded_process(item, excluded_match)]


def _close_cycle_bounds(report: dict[str, Any]) -> tuple[int | None, int | None]:
    plugin = report.get("plugins", {}).get("analysis_queue_delay_decomposition")
    if not isinstance(plugin, dict):
        return None, None
    summary = plugin.get("summary") or {}
    start = summary.get("close_cycle_start_day") if isinstance(summary, dict) else None
    end = summary.get("close_cycle_end_day") if isinstance(summary, dict) else None
    if not (isinstance(start, (int, float)) and isinstance(end, (int, float))):
        for item in plugin.get("findings") or []:
            if not isinstance(item, dict):
                continue
            if "close_cycle_start_day" in item and "close_cycle_end_day" in item:
                start = item.get("close_cycle_start_day")
                end = item.get("close_cycle_end_day")
                break
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return int(start), int(end)
    return None, None


def _dataset_context(
    report: dict[str, Any], storage: Storage
) -> tuple[str | None, str | None, dict[str, str]]:
    lineage = report.get("lineage", {}) or {}
    dataset_version_id = (
        lineage.get("input", {}) or {}
    ).get("dataset_version_id") or (lineage.get("dataset", {}) or {}).get(
        "dataset_version_id"
    )
    if not isinstance(dataset_version_id, str) or not dataset_version_id:
        return None, None, {}
    ctx_row = storage.get_dataset_version_context(dataset_version_id)
    if not ctx_row or not ctx_row.get("table_name"):
        return dataset_version_id, None, {}
    columns = storage.fetch_dataset_columns(dataset_version_id)
    mapping: dict[str, str] = {}
    for col in columns:
        original = col.get("original_name")
        safe = col.get("safe_name")
        role = col.get("role")
        if isinstance(original, str) and original and isinstance(safe, str) and safe:
            mapping[original] = safe
            mapping[original.upper()] = safe
        if isinstance(role, str) and role and isinstance(safe, str) and safe:
            mapping[role.upper()] = safe

    # Add canonical aliases (PROCESS_ID, START_DT, ...) when a dataset template exists.
    dataset_template = storage.fetch_dataset_template(dataset_version_id)
    if isinstance(dataset_template, dict) and str(dataset_template.get("status") or "").strip().lower() == "ready":
        raw_mapping = str(dataset_template.get("mapping_json") or "").strip()
        if raw_mapping:
            try:
                payload = json.loads(raw_mapping)
            except json.JSONDecodeError:
                payload = {}
            mapping_block = payload.get("mapping") if isinstance(payload, dict) else {}
            if isinstance(mapping_block, dict):
                for field_name, meta in mapping_block.items():
                    if not isinstance(field_name, str) or not field_name.strip():
                        continue
                    safe_name = None
                    if isinstance(meta, dict):
                        candidate = meta.get("safe_name")
                        if isinstance(candidate, str) and candidate.strip():
                            safe_name = candidate.strip()
                    if isinstance(safe_name, str) and safe_name:
                        key = field_name.strip()
                        mapping[key] = safe_name
                        mapping[key.upper()] = safe_name
    return dataset_version_id, ctx_row.get("table_name"), mapping


def _sql_positive_numeric_expr(column_name: str) -> str:
    col = str(column_name or "").strip()
    if not col or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", col):
        return "0"
    return f"COALESCE(CAST(NULLIF(TRIM(CAST({col} AS TEXT)), '') AS REAL), 0.0) > 0.0"


def _process_chain_modifiability_map(
    report: dict[str, Any], storage: Storage | None, process_hints: list[str]
) -> dict[str, dict[str, Any]]:
    if storage is None:
        return {}
    normalized = sorted(
        {
            str(p or "").strip().lower()
            for p in process_hints
            if isinstance(p, str) and str(p).strip()
        }
    )
    if not normalized:
        return {}
    _, table_name, mapping = _dataset_context(report, storage)
    if not table_name or not mapping:
        return {}

    process_col = mapping.get("PROCESS_ID")
    master_col = mapping.get("MASTER_PROCESS_QUEUE_ID")
    parent_col = mapping.get("PARENT_PROCESS_QUEUE_ID")
    parent_step_col = mapping.get("PARENT_PROCESS_STEP_QUEUE_ID")
    child_count_col = mapping.get("CHILD_PROCESS_COUNT")
    if not process_col:
        return {}
    child_terms = [
        _sql_positive_numeric_expr(col)
        for col in (master_col, parent_col, parent_step_col)
        if isinstance(col, str) and col.strip()
    ]
    parent_term = (
        _sql_positive_numeric_expr(child_count_col)
        if isinstance(child_count_col, str) and child_count_col.strip()
        else "0"
    )
    if not child_terms and parent_term == "0":
        return {}

    placeholders = ",".join(["?"] * len(normalized))
    child_cond = " OR ".join(f"({term})" for term in child_terms) if child_terms else "0"
    sql = f"""
    SELECT LOWER(TRIM(CAST({process_col} AS TEXT))) AS process_norm,
           COUNT(*) AS runs_total,
           SUM(CASE WHEN ({child_cond}) THEN 1 ELSE 0 END) AS child_link_rows,
           SUM(CASE WHEN ({parent_term}) THEN 1 ELSE 0 END) AS parent_link_rows
    FROM {table_name}
    WHERE LOWER(TRIM(CAST({process_col} AS TEXT))) IN ({placeholders})
    GROUP BY LOWER(TRIM(CAST({process_col} AS TEXT)))
    """
    out: dict[str, dict[str, Any]] = {}
    threshold = _chain_bound_ratio_min()
    try:
        with storage.connection() as conn:
            cur = conn.execute(sql, normalized)
            rows = [dict(r) for r in cur.fetchall()]
    except Exception:
        return {}
    for row in rows:
        process_norm = str(row.get("process_norm") or "").strip().lower()
        if not process_norm:
            continue
        runs_total = int(row.get("runs_total") or 0)
        child_link_rows = int(row.get("child_link_rows") or 0)
        parent_link_rows = int(row.get("parent_link_rows") or 0)
        child_link_ratio = (float(child_link_rows) / float(runs_total)) if runs_total > 0 else 0.0
        parent_link_ratio = (float(parent_link_rows) / float(runs_total)) if runs_total > 0 else 0.0
        chain_bound_ratio = max(child_link_ratio, parent_link_ratio)
        out[process_norm] = {
            "runs_total": runs_total,
            "child_link_rows": child_link_rows,
            "parent_link_rows": parent_link_rows,
            "child_link_ratio": child_link_ratio,
            "parent_link_ratio": parent_link_ratio,
            "chain_bound_ratio": chain_bound_ratio,
            "chain_bound_ratio_threshold": float(threshold),
            "chain_bound": bool(runs_total > 0 and chain_bound_ratio >= float(threshold)),
        }
    return out


def _parse_params(params: str) -> dict[str, str]:
    if not params:
        return {}
    pattern = re.compile(r"\s*([^;=]+?)\s*\([^\)]*\)\s*=\s*([^;]+)")
    out: dict[str, str] = {}
    for match in pattern.finditer(params):
        key = match.group(1).strip().lower()
        value = match.group(2).strip()
        if key:
            out[key] = value
    return out


def _param_overlap_summary(
    storage: Storage,
    report: dict[str, Any],
    process_a: str,
    process_b: str,
) -> dict[str, Any] | None:
    dataset_version_id, table_name, mapping = _dataset_context(report, storage)
    if not table_name or not mapping:
        return None
    proc_col = mapping.get("PROCESS_ID")
    param_col = mapping.get("PARAM_DESCR_LIST")
    start_col = mapping.get("START_DT")
    queue_col = mapping.get("QUEUE_DT")
    if not proc_col or not param_col or (not start_col and not queue_col):
        return None
    start_day, end_day = _close_cycle_bounds(report)
    if not isinstance(start_day, int) or not isinstance(end_day, int):
        start_day = 1
        end_day = 31

    def _day_predicate(col: str, *, start: int, end: int) -> tuple[str, list[Any]]:
        # Works for ISO-like timestamps where day-of-month is at positions 9-10.
        day_expr = f"CAST(SUBSTR({col}, 9, 2) AS INTEGER)"
        if start <= end:
            return f"({day_expr} BETWEEN ? AND ?)", [int(start), int(end)]
        return f"(({day_expr} >= ?) OR ({day_expr} <= ?))", [int(start), int(end)]

    start_pred, start_params = _day_predicate(start_col, start=start_day, end=end_day)
    queue_pred, queue_params = _day_predicate(queue_col, start=start_day, end=end_day)
    placeholders = ",".join(["?"] * 2)
    query = f"""
    SELECT {proc_col} AS process,
           {param_col} AS params,
           {start_col} AS start_dt,
           {queue_col} AS queue_dt
    FROM {table_name}
    WHERE LOWER({proc_col}) IN ({placeholders})
      AND (
            (LENGTH({start_col}) >= 10 AND {start_pred})
         OR (LENGTH({queue_col}) >= 10 AND {queue_pred})
      )
    """
    rows: list[dict[str, Any]] = []
    with storage.connection() as conn:
        cur = conn.execute(
            query,
            [
                process_a.lower(),
                process_b.lower(),
                *start_params,
                *queue_params,
            ],
        )
        rows = [dict(row) for row in cur.fetchall()]

    if not rows:
        return None

    parsed: list[dict[str, Any]] = []
    for row in rows:
        process = str(row.get("process") or "").strip().lower()
        params = _parse_params(row.get("params") or "")
        start_dt = row.get("start_dt") or row.get("queue_dt")
        dt = None
        if isinstance(start_dt, str):
            try:
                dt = datetime.fromisoformat(start_dt.replace("Z", ""))
            except ValueError:
                try:
                    dt = datetime.strptime(start_dt.split(".")[0], "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    dt = None
        parsed.append(
            {
                "process": process,
                "params": params,
                "dt": dt,
            }
        )

    rows_a = [p for p in parsed if p["process"] == process_a.lower() and p["dt"]]
    rows_b = [p for p in parsed if p["process"] == process_b.lower() and p["dt"]]
    if not rows_a or not rows_b:
        return None

    excluded_keys = {
        "business segment",
        "business unit",
        "accounting month",
    }
    keys_a: set[str] = set()
    keys_b: set[str] = set()
    for entry in rows_a:
        keys_a.update(entry.get("params", {}).keys())
    for entry in rows_b:
        keys_b.update(entry.get("params", {}).keys())
    candidate_keys = sorted((keys_a & keys_b) - excluded_keys)
    if not candidate_keys:
        return None

    from collections import Counter

    count_a = len(rows_a)
    count_b = len(rows_b)

    best_summary = None
    best_score = 0.0
    for key in candidate_keys:
        values_a = [p["params"].get(key) for p in rows_a if p["params"].get(key)]
        values_b = [p["params"].get(key) for p in rows_b if p["params"].get(key)]
        if len(set(values_a)) < 2 or len(set(values_b)) < 2:
            continue
        counts_a = Counter(values_a)
        counts_b = Counter(values_b)
        overlap_values = [val for val in counts_a if val in counts_b]
        if not overlap_values:
            continue
        overlap_a = sum(counts_a[val] for val in overlap_values)
        overlap_b = sum(counts_b[val] for val in overlap_values)

        by_value_b: dict[str, list[datetime]] = {}
        for entry in rows_b:
            val = entry.get("params", {}).get(key)
            if not val:
                continue
            by_value_b.setdefault(val, []).append(entry["dt"])
        for val in by_value_b:
            by_value_b[val].sort()

        within_24h = 0
        for entry in rows_a:
            val = entry.get("params", {}).get(key)
            if not val:
                continue
            times = by_value_b.get(val)
            if not times:
                continue
            t = entry["dt"]
            if any(abs((t - other).total_seconds()) <= 86400 for other in times):
                within_24h += 1

        overlap_pct = overlap_a / len(values_a) if values_a else 0.0
        within_pct = within_24h / len(values_a) if values_a else 0.0
        score = overlap_pct * within_pct
        if score > best_score:
            best_score = score
            best_summary = {
                "overlap_key": key,
                "overlap_values": sorted({val for val in overlap_values}),
                "process_a_overlap_rows": overlap_a,
                "process_b_overlap_rows": overlap_b,
                "process_a_overlap_pct": overlap_pct,
                "process_b_overlap_pct": overlap_b / len(values_b) if values_b else 0.0,
                "process_a_within_24h": within_24h,
                "process_a_within_24h_pct": within_pct,
            }

    if not best_summary:
        return None

    return {
        "process_a": process_a,
        "process_b": process_b,
        **best_summary,
        "process_a_rows": count_a,
        "process_b_rows": count_b,
    }


def _metric_context_from_item(
    kind: str | None, item: dict[str, Any], report: dict[str, Any]
) -> dict[str, Any]:
    spec = _metric_spec(kind)
    metric_name = spec.get("name", "Metric")
    return {
        "metric_name": metric_name,
        "definition": spec.get("definition", ""),
        "unit": _metric_unit(metric_name, item),
        "denominator": _denominator_text(item, report, spec),
        "baseline": item.get(spec.get("baseline_field")) if spec.get("baseline_field") else None,
        "observed": item.get(spec.get("observed_field")) if spec.get("observed_field") else None,
        "target_threshold": item.get(spec.get("target_field")) if spec.get("target_field") else None,
    }


def _build_known_issue_recommendations(
    report: dict[str, Any], storage: Storage | None = None, run_dir: Path | None = None
) -> dict[str, Any]:
    known = report.get("known_issues")
    if not isinstance(known, dict):
        return {
            "status": "no_known_issues",
            "summary": "No known issues attached; pass-gate checks skipped.",
            "items": [],
        }
    expected = known.get("expected_findings") or []
    if not isinstance(expected, list) or not expected:
        return {
            "status": "no_expected_findings",
            "summary": "Known issues attached but no expected findings provided.",
            "items": [],
        }

    items: list[dict[str, Any]] = []
    synthetic_match_count = 0
    allow_synthetic_matches = _allow_known_issue_synthetic_matches()
    for issue in expected:
        if not isinstance(issue, dict):
            continue
        plugin_id = issue.get("plugin_id")
        kind = issue.get("kind")
        where = issue.get("where") if isinstance(issue.get("where"), dict) else None
        contains = (
            issue.get("contains") if isinstance(issue.get("contains"), dict) else None
        )
        min_count = issue.get("min_count")
        max_count = issue.get("max_count")
        title = issue.get("title") or issue.get("description") or ""
        if not title:
            label = f"{plugin_id or 'any'}:{kind or 'finding'}"
        else:
            label = title

        findings = _collect_findings_for_plugin(report, plugin_id, kind)
        matched = [f for f in findings if _matches_expected(f, where, contains)]
        count = len(matched)
        evidence_source = "plugin_findings"
        process_hint = _process_hint(where)
        if (
            allow_synthetic_matches
            and count == 0
            and str(kind or "").strip() == "close_cycle_contention"
        ):
            synthetic = _synthetic_close_cycle_contention_finding(
                report, storage, process_hint
            )
            if isinstance(synthetic, dict):
                matched = [synthetic]
                count = 1
                evidence_source = "synthetic_fallback"
                synthetic_match_count += 1

        status = "confirmed"
        if count == 0:
            status = "missing"
        if min_count is not None:
            try:
                if count < int(min_count):
                    status = "below_min"
            except (TypeError, ValueError):
                pass
        if max_count is not None:
            try:
                if count > int(max_count):
                    status = "over_limit"
            except (TypeError, ValueError):
                pass

        recommendation, meta = _capacity_scale_recommendation(
            kind, matched, label, process_hint
        )
        if not recommendation:
            recommendation = _recommendation_text(status, label, process_hint)
            meta = {"action": None, "modeled_delta": None}

        evidence: list[dict[str, Any]] = []
        for item in matched[:3]:
            snippet: dict[str, Any] = {"kind": item.get("kind")}
            for key in ("feature", "pair", "row_index", "index", "score", "metric"):
                if key in item:
                    snippet[key] = item.get(key)
            evidence.append(snippet)

        items.append(
            {
                "title": label,
                "status": status,
                "category": "known",
                "recommendation": recommendation,
                "plugin_id": plugin_id,
                "kind": kind,
                "where": where,
                "contains": contains,
                "expected": {"min_count": min_count, "max_count": max_count},
                "observed_count": count,
                "evidence_source": evidence_source,
                "evidence": evidence,
                "action": meta.get("action"),
                "modeled_delta": meta.get("modeled_delta"),
            }
        )
    deduped = _dedupe_recommendations(items)
    baselines = _duration_baselines(report, storage, run_dir=run_dir)
    qemail_model = _process_removal_model(report, storage, "qemail", run_dir=run_dir)
    deduped = [
        _enrich_recommendation_item(item, report, baselines, qemail_model)
        for item in deduped
        if isinstance(item, dict)
    ]
    failing_count = int(
        sum(
            1
            for item in deduped
            if isinstance(item, dict) and str(item.get("status") or "").lower() != "confirmed"
        )
    )
    status = "failing" if failing_count > 0 else "ok"
    summary = f"Generated {len(deduped)} recommendation(s) from known issues."
    if failing_count > 0:
        summary += f" {failing_count} landmark(s) are not independently confirmed."
    if synthetic_match_count > 0:
        summary += (
            f" Synthetic fallback matches used: {synthetic_match_count} "
            f"(diagnostic-only; controlled by {_ALLOW_KNOWN_ISSUE_SYNTHETIC_MATCHES_ENV}=1)."
        )
    return {
        "status": status,
        "summary": summary,
        "items": deduped,
    }


def _build_discovery_recommendations(
    report: dict[str, Any], storage: Storage | None = None, run_dir: Path | None = None
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    # Discovery recommendations should remain visible even when a process is
    # referenced by known-issue gates; otherwise we hide the "why" behind a
    # generic pass/fail label. Only honor explicit exclusions.
    excluded_processes = _explicit_excluded_processes(report)
    excluded_match = compile_patterns(sorted(excluded_processes)) if excluded_processes else None
    plugins = report.get("plugins", {}) or {}

    queue_plugin = plugins.get("analysis_queue_delay_decomposition")
    ops_plugin = plugins.get("analysis_actionable_ops_levers_v1")
    ops_by_process: dict[str, list[dict[str, Any]]] = {}
    if isinstance(ops_plugin, dict):
        for f in ops_plugin.get("findings") or []:
            if not isinstance(f, dict) or f.get("kind") != "actionable_ops_lever":
                continue
            proc = f.get("process_norm") or f.get("process") or ""
            if isinstance(proc, str) and proc.strip():
                ops_by_process.setdefault(proc.strip().lower(), []).append(f)
        for k, rows in list(ops_by_process.items()):
            ops_by_process[k] = sorted(
                rows,
                key=lambda r: float(r.get("expected_delta_seconds") or 0.0),
                reverse=True,
            )
    related_processes: set[str] = set()
    shift_processes: set[str] = set()
    overlap_cache: dict[tuple[str, str], dict[str, Any] | None] = {}
    if isinstance(queue_plugin, dict):
        findings = queue_plugin.get("findings") or []
        impact = next(
            (f for f in findings if isinstance(f, dict) and f.get("kind") == "eligible_wait_impact"),
            None,
        )
        total_gt = (
            float(impact.get("eligible_wait_gt_hours_total"))
            if isinstance(impact, dict)
            and isinstance(impact.get("eligible_wait_gt_hours_total"), (int, float))
            else None
        )
        process_stats = [
            f for f in findings if isinstance(f, dict) and f.get("kind") == "eligible_wait_process_stats"
        ]
        process_stats = sorted(
            process_stats,
            key=lambda row: float(row.get("eligible_wait_gt_hours_total") or 0.0),
            reverse=True,
        )
        max_queue_recs = 3
        queue_recs = 0
        for idx, item in enumerate(process_stats):
            process = item.get("process_norm") or item.get("process")
            if not isinstance(process, str) or not process.strip():
                continue
            if idx < 10:
                related_processes.add(process.strip().lower())
            process_key = process.strip().lower()
            if excluded_match and excluded_match(process_key):
                continue
            value = item.get("eligible_wait_gt_hours_total")
            if not isinstance(value, (int, float)) or float(value) < 1.0:
                continue
            runs_total = item.get("runs_total")
            if isinstance(runs_total, (int, float)) and float(runs_total) < 500:
                continue
            share = None
            if isinstance(total_gt, (int, float)) and total_gt > 0:
                share = (float(value) / float(total_gt)) * 100.0
            context = _metric_context_from_item("eligible_wait_process_stats", item, report)
            evidence = {
                "process": process,
                "wait_beyond_target_hours_total": float(value),
                "share_percent": float(share) if share is not None else None,
                "runs_total": runs_total,
            }
            levers = ops_by_process.get(process_key) or []
            lever_summaries: list[str] = []
            lever_validation: list[str] = []
            lever_evidence_paths: list[str] = []
            for lever in levers[:2]:
                title = str(lever.get("title") or "").strip()
                if not title:
                    continue
                delta_s = lever.get("expected_delta_seconds")
                delta_txt = ""
                if isinstance(delta_s, (int, float)) and float(delta_s) > 0:
                    delta_txt = f" (upper bound ~{float(delta_s)/3600.0:.1f}h total over the observation window)"
                lever_summaries.append(title + delta_txt)
                vsteps = lever.get("validation_steps")
                if isinstance(vsteps, list):
                    for s in vsteps:
                        if isinstance(s, str) and s.strip():
                            lever_validation.append(s.strip())
                ev = lever.get("evidence") if isinstance(lever.get("evidence"), dict) else {}
                for v in ev.values():
                    if isinstance(v, str) and v.endswith(".json"):
                        lever_evidence_paths.append(v)
            if lever_summaries:
                evidence["ops_levers"] = lever_summaries
            if lever_evidence_paths:
                evidence["ops_evidence_paths"] = sorted(set(lever_evidence_paths))[:8]
            rec_text = (
                f"Process {process} has about {float(value):.2f} hours of wait time beyond the target"
                + (f" ({share:.1f}% of all such wait time)" if share is not None else "")
                + (f" across {int(runs_total):,} runs." if isinstance(runs_total, (int, float)) else ".")
            )
            if lever_summaries:
                rec_text += " Suggested levers: " + "; ".join(lever_summaries) + "."
            else:
                rec_text += " Next step: review the process-specific levers output and look for routing, scheduling, and parameter-bucketing actions."
            confidence_weight = _confidence_weight(item, {})
            controllability_weight = _controllability_weight(
                "eligible_wait_process_stats", {}
            )
            impact_hours = float(value)
            relevance_score = impact_hours * confidence_weight * controllability_weight
            items.append(
                {
                    "title": f"Reduce over-threshold wait for {process}",
                    "status": "discovered",
                    "category": "discovery",
                    "recommendation": rec_text,
                    "plugin_id": "analysis_queue_delay_decomposition",
                    "kind": "eligible_wait_process_stats",
                    "where": {"process_norm": process},
                    "contains": None,
                    "observed_count": item.get("runs_total"),
                    "evidence": [evidence],
                    "validation_steps": sorted(set(lever_validation))[:6] if lever_validation else None,
                    "action": "reduce_process_wait",
                    "modeled_delta": None,
                    "measurement_type": item.get("measurement_type") or "measured",
                    "impact_hours": impact_hours,
                    "confidence_weight": confidence_weight,
                    "controllability_weight": controllability_weight,
                    "relevance_score": relevance_score,
                    **context,
                }
            )
            queue_recs += 1
            if queue_recs >= max_queue_recs:
                break

        # If overall wait exists but over-threshold wait is exactly 0, the
        # configured thresholding predicate is likely suppressing signal.
        if (
            isinstance(queue_plugin.get("metrics"), dict)
            and isinstance(queue_plugin["metrics"].get("eligible_wait_hours_total"), (int, float))
            and isinstance(queue_plugin["metrics"].get("eligible_wait_gt_hours_total"), (int, float))
        ):
            wait_total = float(queue_plugin["metrics"].get("eligible_wait_hours_total") or 0.0)
            wait_gt_total = float(queue_plugin["metrics"].get("eligible_wait_gt_hours_total") or 0.0)
            if wait_total > 0.0 and wait_gt_total == 0.0:
                rec_text = (
                    "Over-threshold wait is 0.0h while total eligible wait is "
                    f"{wait_total:.2f}h. This often means the wait thresholding is set at or above "
                    "nearly all observed waits (and the predicate is strict `>`). "
                    "If you expect a busy-period signal, lower the threshold slightly or switch to `>=`."
                )
                evidence = {
                    "eligible_wait_hours_total": wait_total,
                    "eligible_wait_gt_hours_total": wait_gt_total,
                    "standalone_runs": (queue_plugin.get("metrics") or {}).get("standalone_runs"),
                    "sequence_runs": (queue_plugin.get("metrics") or {}).get("sequence_runs"),
                }
                confidence_weight = 0.9
                controllability_weight = 0.9
                impact_hours = wait_total
                relevance_score = impact_hours * confidence_weight * controllability_weight
                items.append(
                    {
                        "title": "Revisit over-threshold wait thresholding",
                        "status": "discovered",
                        "category": "discovery",
                        "recommendation": rec_text,
                        "plugin_id": "analysis_queue_delay_decomposition",
                        "kind": "eligible_wait_impact",
                        "where": None,
                        "contains": None,
                        "observed_count": (queue_plugin.get("metrics") or {}).get("standalone_runs"),
                        "evidence": [evidence],
                        "action": "tune_threshold",
                        "modeled_delta": None,
                        "measurement_type": "measured",
                        "impact_hours": impact_hours,
                        "confidence_weight": confidence_weight,
                        "controllability_weight": controllability_weight,
                        "relevance_score": relevance_score,
                    }
                )

    # Sequence bottlenecks: process-to-process handoff gaps that dominate avoidable waiting.
    seq_plugin = plugins.get("analysis_process_sequence_bottlenecks")
    if isinstance(seq_plugin, dict):
        seq_findings = [
            f
            for f in (seq_plugin.get("findings") or [])
            if isinstance(f, dict) and f.get("kind") == "sequence_bottleneck"
        ]
        seq_findings = sorted(
            seq_findings,
            key=lambda row: float(row.get("relevance_score") or 0.0),
            reverse=True,
        )
        results_paths = _artifact_paths(report, "analysis_process_sequence_bottlenecks")
        results_path = next((p for p in results_paths if p.endswith("results.json")), None)
        for item in seq_findings[:5]:
            transition = item.get("transition")
            if not isinstance(transition, str) or not transition.strip():
                continue
            delta_hours = item.get("delta_hours")
            baseline_hours = item.get("baseline_over_threshold_hours")
            modeled_hours = item.get("modeled_over_threshold_hours")
            if not isinstance(delta_hours, (int, float)) or float(delta_hours) <= 0:
                continue
            evidence_cases: list[str] = []
            ev = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
            examples = ev.get("example_cases")
            if isinstance(examples, list):
                evidence_cases = [str(x) for x in examples if str(x).strip()][:3]

            rec_text = (
                f"Handoff gap {transition} is a concentrated source of waiting beyond the target. "
                f"Estimated avoidable wait: ~{float(delta_hours):.2f} hours (upper bound) if this handoff behaves like its baseline. "
                "Action ideas: align the downstream schedule to the upstream completion window; remove batching between the two steps; "
                "or add a trigger so the downstream step starts immediately after the upstream finishes."
            )
            if isinstance(baseline_hours, (int, float)) and isinstance(modeled_hours, (int, float)):
                rec_text += f" (Current ~{float(baseline_hours):.2f}h -> baseline ~{float(modeled_hours):.2f}h.)"
            if evidence_cases:
                rec_text += f" Example case IDs: {', '.join(evidence_cases)}."

            confidence_weight = _confidence_weight(item, {})
            controllability_weight = 0.6
            impact_hours = float(delta_hours)
            relevance_score = impact_hours * float(confidence_weight) * controllability_weight
            context = _metric_context_from_item("sequence_bottleneck", item, report)
            evidence = {
                "transition": transition,
                "delta_hours": float(delta_hours),
                "baseline_over_threshold_hours": float(baseline_hours) if isinstance(baseline_hours, (int, float)) else None,
                "modeled_over_threshold_hours": float(modeled_hours) if isinstance(modeled_hours, (int, float)) else None,
                "transition_count": item.get("transition_count"),
                "example_cases": evidence_cases,
                "results_artifact": results_path,
                "columns": item.get("columns"),
            }
            items.append(
                {
                    "title": f"Reduce handoff gap {transition}",
                    "status": "discovered",
                    "category": "discovery",
                    "recommendation": rec_text,
                    "plugin_id": "analysis_process_sequence_bottlenecks",
                    "kind": "sequence_bottleneck",
                    "where": {"transition": transition},
                    "contains": None,
                    "observed_count": item.get("transition_count"),
                    "evidence": [evidence],
                    "action": "reduce_transition_gap",
                    "action_type": "reduce_transition_gap",
                    "target": transition,
                    "scenario_id": item.get("scenario_id") or None,
                    "modeled_delta": float(delta_hours),
                    "measurement_type": item.get("measurement_type") or "modeled",
                    "impact_hours": impact_hours,
                    "confidence_weight": float(confidence_weight),
                    "controllability_weight": controllability_weight,
                    "relevance_score": float(item.get("relevance_score") or relevance_score),
                    "validation_steps": [
                        "Pick a few example case IDs and confirm the downstream step is not starting immediately after upstream completion.",
                        "Apply a targeted scheduling/trigger change for that handoff.",
                        "Re-run the harness and verify the handoff gap shrinks without shifting the bottleneck elsewhere.",
                    ],
                    **context,
                }
            )

    # Fallback: if queue-delay decomposition is unavailable/skipped, derive actionable "top driver"
    # recommendations directly from busy-period segmentation v2.
    if not items and run_dir is not None:
        seg_plugin = plugins.get("analysis_busy_period_segmentation_v2")
        seg_path = None
        if isinstance(seg_plugin, dict):
            for artifact in seg_plugin.get("artifacts") or []:
                if not isinstance(artifact, dict):
                    continue
                path = artifact.get("path")
                if isinstance(path, str) and path.endswith("busy_periods.json"):
                    seg_path = path
                    break
        seg_payload = _load_artifact_json(run_dir, seg_path)
        busy_periods = (
            seg_payload.get("busy_periods")
            if isinstance(seg_payload, dict)
            else None
        )
        if isinstance(busy_periods, list) and busy_periods:
            process_totals: dict[str, float] = {}
            process_runs_over: dict[str, int] = {}
            top_period_for_proc: dict[str, dict[str, Any]] = {}
            for bp in busy_periods:
                if not isinstance(bp, dict):
                    continue
                per_process = bp.get("per_process_over_threshold_wait_sec")
                if not isinstance(per_process, dict):
                    continue
                runs_over = int(bp.get("runs_over_threshold_count") or 0)
                for proc, sec in per_process.items():
                    try:
                        proc_key = str(proc)
                        sec_f = float(sec)
                    except (TypeError, ValueError):
                        continue
                    process_totals[proc_key] = process_totals.get(proc_key, 0.0) + sec_f
                    if runs_over:
                        process_runs_over[proc_key] = process_runs_over.get(proc_key, 0) + runs_over
                    prev = top_period_for_proc.get(proc_key)
                    if prev is None or float(prev.get("wait_sec") or 0.0) < sec_f:
                        top_period_for_proc[proc_key] = {
                            "busy_period_id": bp.get("busy_period_id"),
                            "start_ts": bp.get("start_ts"),
                            "end_ts": bp.get("end_ts"),
                            "wait_sec": sec_f,
                        }

            total_sec = sum(process_totals.values()) or 0.0
            ranked = sorted(process_totals.items(), key=lambda kv: kv[1], reverse=True)
            for proc, sec_total in ranked[:3]:
                proc_key = proc.strip().lower()
                if excluded_match and excluded_match(proc_key):
                    continue
                hours = float(sec_total) / 3600.0
                if hours < 1.0:
                    continue
                share = (float(sec_total) / float(total_sec) * 100.0) if total_sec else None
                period = top_period_for_proc.get(proc) or {}
                evidence = {
                    "process": proc,
                    "wait_to_start_gt_hours_total": round(hours, 2),
                    "share_percent": round(float(share), 2) if share is not None else None,
                    "runs_over_threshold_count_sum": int(process_runs_over.get(proc, 0)),
                    "top_busy_period": period,
                }
                rec_text = (
                    f"Reduce >threshold wait-to-start for {proc}: {hours:.2f}h"
                    + (f" ({share:.1f}% of total)" if share is not None else "")
                    + (f"; top busy period {period.get('busy_period_id')} {period.get('start_ts')}..{period.get('end_ts')}." if period.get("busy_period_id") else ".")
                )
                confidence_weight = 0.8
                controllability_weight = 0.7
                impact_hours = hours
                relevance_score = impact_hours * confidence_weight * controllability_weight
                items.append(
                    {
                        "title": f"Reduce over-threshold wait-to-start for {proc}",
                        "status": "discovered",
                        "category": "discovery",
                        "recommendation": rec_text,
                        "plugin_id": "analysis_busy_period_segmentation_v2",
                        "kind": "busy_periods_top_driver",
                        "where": {"process": proc},
                        "contains": None,
                        "observed_count": int(process_runs_over.get(proc, 0)) or None,
                        "evidence": [evidence],
                        "action": "reduce_process_wait",
                        "modeled_delta": None,
                        "measurement_type": "measured",
                        "impact_hours": impact_hours,
                        "confidence_weight": confidence_weight,
                        "controllability_weight": controllability_weight,
                        "relevance_score": relevance_score,
                    }
                )

    # Note: a previous discovery recommendation suggested adding trace/dependency keys.
    # For many ERP systems, that is not operationally feasible. Keep this suppressed
    # unless explicitly enabled.
    allow_data_instrumentation = os.environ.get(
        "STAT_HARNESS_ALLOW_DATA_INSTRUMENTATION", ""
    ).lower() in {"1", "true", "yes"}
    if allow_data_instrumentation:
        seq_cls_plugin = plugins.get("analysis_sequence_classification")
        proc_seq_plugin = plugins.get("analysis_process_sequence")
        dep_join_plugin = plugins.get("analysis_dependency_resolution_join")
        try:
            seq_metrics = seq_cls_plugin.get("metrics") if isinstance(seq_cls_plugin, dict) else {}
            proc_metrics = proc_seq_plugin.get("metrics") if isinstance(proc_seq_plugin, dict) else {}
            dep_metrics = dep_join_plugin.get("metrics") if isinstance(dep_join_plugin, dict) else {}
            standalone_runs = float(seq_metrics.get("standalone_runs") or 0.0) if isinstance(seq_metrics, dict) else 0.0
            sequence_runs = float(seq_metrics.get("sequence_runs") or 0.0) if isinstance(seq_metrics, dict) else 0.0
            variants = float(proc_metrics.get("variants") or 0.0) if isinstance(proc_metrics, dict) else 0.0
            transitions = float(proc_metrics.get("transitions") or 0.0) if isinstance(proc_metrics, dict) else 0.0
            dependency_rows = float(dep_metrics.get("dependency_rows") or 0.0) if isinstance(dep_metrics, dict) else 0.0
            missing_trace = (
                standalone_runs >= 20.0
                and sequence_runs == 0.0
                and variants >= 2.0
                and transitions == 0.0
            )
            if missing_trace:
                rec_text = (
                    "This dataset looks like an eventlog (multiple variants) but has no sequenced runs "
                    "(sequence_runs=0, transitions=0). That usually means there is no usable trace key "
                    "or dependency column linking events across steps."
                )
                evidence = {
                    "standalone_runs": int(standalone_runs),
                    "sequence_runs": int(sequence_runs),
                    "variants": int(variants),
                    "transitions": int(transitions),
                    "dependency_rows": int(dependency_rows),
                    "sequence_summary": seq_cls_plugin.get("summary") if isinstance(seq_cls_plugin, dict) else None,
                    "dependency_summary": dep_join_plugin.get("summary") if isinstance(dep_join_plugin, dict) else None,
                }
                confidence_weight = 0.8
                controllability_weight = 0.0
                impact_hours = 0.0
                relevance_score = confidence_weight * controllability_weight
                items.append(
                    {
                        "title": "Missing trace/dependency keys limit cross-step bottleneck insights",
                        "status": "discovered",
                        "category": "discovery",
                        "recommendation": rec_text,
                        "plugin_id": "analysis_sequence_classification",
                        "kind": "sequence_not_detected",
                        "where": None,
                        "contains": None,
                        "observed_count": int(standalone_runs),
                        "evidence": [evidence],
                        "action": "blocked_by_missing_trace_keys",
                        "modeled_delta": None,
                        "measurement_type": "measured",
                        "impact_hours": impact_hours,
                        "confidence_weight": confidence_weight,
                        "controllability_weight": controllability_weight,
                        "relevance_score": relevance_score,
                    }
                )
        except Exception:
            pass

    contention_plugin = plugins.get("analysis_close_cycle_contention")
    if isinstance(contention_plugin, dict):
        findings = [
            f
            for f in contention_plugin.get("findings") or []
            if isinstance(f, dict) and f.get("kind") == "close_cycle_contention"
        ]
        findings = sorted(
            findings,
            key=lambda row: float(row.get("estimated_improvement_pct") or 0.0)
            * float(row.get("close_count") or 0.0),
            reverse=True,
        )
        for item in findings[:3]:
            process = item.get("process_norm") or item.get("process")
            if not isinstance(process, str) or not process.strip():
                continue
            process_key = process.strip().lower()
            if excluded_match and excluded_match(process_key):
                continue
            related_processes.add(process_key)

            slowdown = item.get("slowdown_ratio")
            correlation = item.get("correlation")
            improvement_pct = item.get("estimated_improvement_pct")
            modeled_reduction_pct = item.get("modeled_reduction_pct")
            modeled_reduction_hours = item.get("modeled_reduction_hours")
            close_count = item.get("close_count")
            open_count = item.get("open_count")
            median_close = item.get("median_duration_close")
            median_open = item.get("median_duration_open")
            servers = item.get("servers") or []

            # Approximate impact: how much extra time "other work" spends during
            # close cycle when this process is present.
            impact_hours = 0.0
            if isinstance(median_close, (int, float)) and isinstance(median_open, (int, float)) and isinstance(close_count, (int, float)):
                impact_hours = (max(0.0, float(median_close) - float(median_open)) * float(close_count)) / 3600.0

            context = _metric_context_from_item("close_cycle_contention", item, report)
            confidence_weight = _confidence_weight(item, {})
            controllability_weight = _controllability_weight(
                "close_cycle_contention", {}
            )
            relevance_score = impact_hours * confidence_weight * controllability_weight

            server_text = ""
            if isinstance(servers, list) and servers:
                server_text = f" (top servers: {', '.join(str(s) for s in servers[:3])})"

            rec_text = (
                f"Isolate or reschedule {process} during close cycle: other-work median duration "
                f"{median_close}s vs {median_open}s ({slowdown:.2f}x; corr={correlation:.2f}){server_text}. "
                f"Close/open counts: {int(close_count) if isinstance(close_count, (int, float)) else 'n/a'}/"
                f"{int(open_count) if isinstance(open_count, (int, float)) else 'n/a'}. "
                f"Estimated improvement if removed/isolated: {float(improvement_pct)*100.0:.1f}%."
            )
            if isinstance(modeled_reduction_pct, (int, float)) and float(modeled_reduction_pct) > 0.0:
                rec_text += f" Modeled close-window reduction from removing this process: {float(modeled_reduction_pct) * 100.0:.1f}%."
            if isinstance(modeled_reduction_hours, (int, float)) and float(modeled_reduction_hours) > 0.0:
                rec_text += f" Modeled reduction hours: {float(modeled_reduction_hours):.2f}h."
            evidence = {
                "process": process,
                "slowdown_ratio": slowdown,
                "correlation": correlation,
                "median_duration_close": median_close,
                "median_duration_open": median_open,
                "estimated_improvement_pct": improvement_pct,
                "modeled_reduction_pct": modeled_reduction_pct,
                "modeled_reduction_hours": modeled_reduction_hours,
                "close_count": close_count,
                "open_count": open_count,
                "servers": servers[:5] if isinstance(servers, list) else [],
            }
            items.append(
                {
                    "title": f"Isolate close-cycle contention for {process}",
                    "status": "discovered",
                    "category": "discovery",
                    "recommendation": rec_text,
                    "plugin_id": "analysis_close_cycle_contention",
                    "kind": "close_cycle_contention",
                    "where": {"process_norm": process},
                    "contains": None,
                    "observed_count": close_count,
                    "evidence": [evidence],
                    "action": "isolate_process",
                    "modeled_delta": None,
                    "measurement_type": item.get("measurement_type") or "measured",
                    "impact_hours": impact_hours,
                    "confidence_weight": confidence_weight,
                    "controllability_weight": controllability_weight,
                    "relevance_score": relevance_score,
                    **context,
                }
            )

    shift_plugin = plugins.get("analysis_close_cycle_duration_shift")
    if isinstance(shift_plugin, dict):
        findings = [
            f
            for f in shift_plugin.get("findings") or []
            if isinstance(f, dict) and f.get("kind") == "close_cycle_duration_shift"
        ]
        candidates: list[dict[str, Any]] = []
        for item in findings:
            process = item.get("process_norm") or item.get("process")
            if not isinstance(process, str) or not process.strip():
                continue
            shift_processes.add(process.strip().lower())
            related_processes.add(process.strip().lower())
            process_key = process.strip().lower()
            if excluded_match and excluded_match(process_key):
                continue
            p_value = item.get("p_value")
            slowdown = item.get("slowdown_ratio")
            effect = item.get("effect_seconds")
            close_count = item.get("close_count")
            if not (isinstance(p_value, (int, float)) and p_value <= 0.05):
                continue
            if not (isinstance(slowdown, (int, float)) and slowdown >= 1.2):
                continue
            if not (isinstance(effect, (int, float)) and effect >= 5.0):
                continue
            if not (isinstance(close_count, (int, float)) and close_count >= 200):
                continue
            candidates.append(item)
        candidates = sorted(
            candidates,
            key=lambda row: float(row.get("effect_seconds") or 0.0)
            * float(row.get("close_count") or 0.0),
            reverse=True,
        )
        for item in candidates[:2]:
            process = item.get("process_norm") or item.get("process")
            median_close = item.get("median_close")
            median_open = item.get("median_open")
            effect = float(item.get("effect_seconds") or 0.0)
            slowdown = float(item.get("slowdown_ratio") or 0.0)
            p_value = item.get("p_value")
            close_count = float(item.get("close_count") or 0.0)
            impact_hours = (effect * close_count) / 3600.0
            context = _metric_context_from_item("close_cycle_duration_shift", item, report)
            rec_text = (
                f"Reduce close-cycle slowdown for {process}: median {median_close}s vs "
                f"{median_open}s (Δ {effect:.1f}s, {slowdown:.2f}x; p={p_value:.4g}). "
                "Reschedule or add dedicated capacity during close cycle."
            )
            overlap_detail = None
            if storage and related_processes:
                best = None
                candidate_pool = sorted(shift_processes - {process.strip().lower()})
                if not candidate_pool:
                    candidate_pool = sorted(related_processes - {process.strip().lower()})
                candidate_pool = candidate_pool[:5]
                for other in candidate_pool:
                    if other == process.strip().lower():
                        continue
                    key = tuple(sorted([process.strip().lower(), other]))
                    if key in overlap_cache:
                        summary = overlap_cache[key]
                    else:
                        summary = _param_overlap_summary(storage, report, process, other)
                        overlap_cache[key] = summary
                    if not summary:
                        continue
                    score = (
                        float(summary.get("process_a_overlap_pct") or 0.0)
                        * float(summary.get("process_a_within_24h_pct") or 0.0)
                    )
                    if not best or score > best[0]:
                        best = (score, summary)
                    if (
                        summary.get("process_a_overlap_pct", 0.0) >= 0.9
                        and summary.get("process_a_within_24h_pct", 0.0) >= 0.8
                    ):
                        break
                if best and best[1]:
                    overlap_detail = best[1]
                    if (
                        overlap_detail.get("process_a_overlap_pct", 0.0) >= 0.7
                        and overlap_detail.get("process_a_within_24h_pct", 0.0) >= 0.7
                    ):
                        overlap_key = overlap_detail.get("overlap_key") or "parameters"
                        overlap_values = ", ".join(
                            overlap_detail.get("overlap_values") or []
                        )
                        rec_text = (
                            rec_text
                            + " Observed overlapping parameters with "
                            + f"{overlap_detail.get('process_b')} on {overlap_key} {overlap_values}; "
                            + f"{overlap_detail.get('process_a_within_24h_pct', 0.0)*100:.1f}% run within 24h. "
                            + "Add a dependency so this process runs after the matching parameter set completes."
                        )
            confidence_weight = _confidence_weight(item, {})
            controllability_weight = _controllability_weight(
                "close_cycle_duration_shift", {}
            )
            relevance_score = impact_hours * confidence_weight * controllability_weight
            evidence = {
                "process": process,
                "median_close": median_close,
                "median_open": median_open,
                "effect_seconds": effect,
                "slowdown_ratio": slowdown,
                "p_value": p_value,
                "close_count": close_count,
                "open_count": item.get("open_count"),
            }
            if overlap_detail:
                evidence.update(
                    {
                        "related_process": overlap_detail.get("process_b"),
                        "overlap_key": overlap_detail.get("overlap_key"),
                        "overlap_values": overlap_detail.get("overlap_values"),
                        "overlap_pct": overlap_detail.get("process_a_overlap_pct"),
                        "within_24h_pct": overlap_detail.get("process_a_within_24h_pct"),
                    }
                )
            items.append(
                {
                    "title": f"Reduce close-cycle slowdown for {process}",
                    "status": "discovered",
                    "category": "discovery",
                    "recommendation": rec_text,
                    "plugin_id": "analysis_close_cycle_duration_shift",
                    "kind": "close_cycle_duration_shift",
                    "where": {"process_norm": process},
                    "contains": None,
                    "observed_count": item.get("close_count"),
                    "evidence": [evidence],
                    "action": "reduce_close_cycle_slowdown",
                    "modeled_delta": None,
                    "measurement_type": item.get("measurement_type") or "measured",
                    "impact_hours": impact_hours,
                    "confidence_weight": confidence_weight,
                    "controllability_weight": controllability_weight,
                    "relevance_score": relevance_score,
                    **context,
                }
            )

    # Classic stats auto-scan: surface non-obvious operational levers beyond
    # known landmarks (server/module/params levels that materially change
    # duration for a specific process).
    anova_plugin = plugins.get("analysis_anova_auto")
    if isinstance(anova_plugin, dict):
        findings = [
            f
            for f in anova_plugin.get("findings") or []
            if isinstance(f, dict) and f.get("kind") == "anova_process_group_effect"
        ]
        findings = sorted(
            findings,
            key=lambda row: float(row.get("delta_median") or 0.0)
            * float(row.get("n") or 0.0),
            reverse=True,
        )
        added = 0
        for item in findings:
            if added >= 5:
                break
            process = item.get("process_norm") or item.get("process")
            if not isinstance(process, str) or not process.strip() or process.strip().lower() in {"all"}:
                continue
            process_key = process.strip().lower()
            if excluded_match and excluded_match(process_key):
                continue

            delta = item.get("delta_median")
            n = item.get("n")
            q = item.get("q_value")
            group_col = item.get("group_column")
            worst = item.get("worst_level")
            best = item.get("best_level")
            metric = item.get("metric")
            if not (isinstance(delta, (int, float)) and isinstance(n, (int, float))):
                continue
            if not (isinstance(metric, str) and metric):
                continue
            if not (isinstance(group_col, str) and group_col):
                continue

            impact_hours = (abs(float(delta)) * float(n)) / 3600.0
            confidence_weight = 0.8 if isinstance(q, (int, float)) and float(q) <= 0.05 else 0.6
            controllability_weight = 0.9
            relevance_score = impact_hours * confidence_weight * controllability_weight

            rec_text = (
                f"Rebalance {process} across {group_col}: '{worst}' median {metric} is ~{abs(float(delta)):.2f} "
                f"worse than '{best}'. Shift close-window load away from '{worst}' and validate median improvement."
            )
            evidence = {
                "process": process,
                "group_column": group_col,
                "metric": metric,
                "worst_level": worst,
                "best_level": best,
                "delta_median": float(delta),
                "n": int(n),
                "q_value": float(q) if isinstance(q, (int, float)) else None,
                "levels": item.get("levels"),
                "level_medians": item.get("level_medians"),
            }
            items.append(
                {
                    "title": f"Rebalance {process} across {group_col}",
                    "status": "discovered",
                    "category": "discovery",
                    "recommendation": rec_text,
                    "plugin_id": "analysis_anova_auto",
                    "kind": "anova_process_group_effect",
                    "where": {"process_norm": process},
                    "contains": None,
                    "observed_count": int(n),
                    "evidence": [evidence],
                    "action": "rebalance_group",
                    "modeled_delta": None,
                    "measurement_type": item.get("measurement_type") or "measured",
                    "impact_hours": impact_hours,
                    "confidence_weight": confidence_weight,
                    "controllability_weight": controllability_weight,
                    "relevance_score": relevance_score,
                }
            )
            added += 1

    counter_plugin = plugins.get("analysis_process_counterfactuals")
    if isinstance(counter_plugin, dict):
        findings = [
            f
            for f in counter_plugin.get("findings") or []
            if isinstance(f, dict) and f.get("kind") == "process_counterfactual"
        ]
        for item in findings:
            process = item.get("process_norm") or item.get("process_id")
            if not isinstance(process, str) or not process.strip():
                continue
                if excluded_match and excluded_match(process.strip()):
                    continue
            delta_hours = item.get("delta_hours") or item.get("delta_value")
            if not isinstance(delta_hours, (int, float)) or float(delta_hours) <= 0:
                continue
            baseline_hours = item.get("baseline_over_threshold_hours")
            modeled_hours = item.get("modeled_over_threshold_hours")
            runs_count = item.get("runs_count")
            context = _metric_context_from_item("process_counterfactual", item, report)
            confidence_weight = item.get("confidence_weight")
            if not isinstance(confidence_weight, (int, float)):
                confidence_weight = _confidence_weight(item, {})
            controllability_weight = item.get("controllability_weight")
            if not isinstance(controllability_weight, (int, float)):
                controllability_weight = _controllability_weight(
                    "process_counterfactual", {}
                )
            impact_hours = float(delta_hours)
            relevance_score = (
                impact_hours * float(confidence_weight) * float(controllability_weight)
            )
            evidence = {
                "process": process,
                "baseline_hours": baseline_hours,
                "modeled_hours": modeled_hours,
                "delta_hours": delta_hours,
                "runs_count": runs_count,
                "columns": item.get("columns"),
            }
            items.append(
                {
                    "title": f"Reduce over-threshold wait for {process}",
                    "status": "discovered",
                    "category": "discovery",
                    "recommendation": item.get("recommendation")
                    or f"Reduce >threshold wait for {process} (Δ {float(delta_hours):.2f}h).",
                    "plugin_id": "analysis_process_counterfactuals",
                    "kind": "process_counterfactual",
                    "where": {"process_norm": process},
                    "contains": None,
                    "observed_count": runs_count,
                    "evidence": [evidence],
                    "action": item.get("action_type") or "reduce_process_wait",
                    "action_type": item.get("action_type") or "reduce_process_wait",
                    "target": item.get("target") or process,
                    "scenario_id": item.get("scenario_id"),
                    "delta_signature": item.get("delta_signature"),
                    "modeled_delta": float(delta_hours),
                    "measurement_type": item.get("measurement_type") or "modeled",
                    "impact_hours": impact_hours,
                    "confidence_weight": confidence_weight,
                    "controllability_weight": controllability_weight,
                    "relevance_score": relevance_score,
                    **context,
                }
            )

    bottleneck_plugin = plugins.get("analysis_process_sequence_bottlenecks")
    if isinstance(bottleneck_plugin, dict):
        findings = [
            f
            for f in bottleneck_plugin.get("findings") or []
            if isinstance(f, dict) and f.get("kind") == "sequence_bottleneck"
        ]
        for item in findings:
            delta_hours = item.get("delta_hours") or item.get("delta_value")
            if not isinstance(delta_hours, (int, float)) or float(delta_hours) <= 0:
                continue
            process = item.get("process_id")
            next_process = item.get("next_process_id")
            transition = item.get("transition") or ""
            if not isinstance(transition, str) or not transition.strip():
                transition = f"{process} -> {next_process}"
            context = _metric_context_from_item("sequence_bottleneck", item, report)
            confidence_weight = item.get("confidence_weight")
            if not isinstance(confidence_weight, (int, float)):
                confidence_weight = _confidence_weight(item, {})
            controllability_weight = item.get("controllability_weight")
            if not isinstance(controllability_weight, (int, float)):
                controllability_weight = _controllability_weight(
                    "sequence_bottleneck", {}
                )
            impact_hours = float(delta_hours)
            relevance_score = (
                impact_hours * float(confidence_weight) * float(controllability_weight)
            )
            evidence = {
                "transition": transition,
                "baseline_hours": item.get("baseline_over_threshold_hours"),
                "modeled_hours": item.get("modeled_over_threshold_hours"),
                "delta_hours": delta_hours,
                "transition_count": item.get("transition_count"),
                "columns": item.get("columns"),
            }
            items.append(
                {
                    "title": f"Reduce handoff gap {transition}",
                    "status": "discovered",
                    "category": "discovery",
                    "recommendation": item.get("recommendation")
                    or f"Reduce transition gap {transition} (Δ {float(delta_hours):.2f}h).",
                    "plugin_id": "analysis_process_sequence_bottlenecks",
                    "kind": "sequence_bottleneck",
                    "where": {
                        "process_norm": process,
                        "next_process_norm": next_process,
                    },
                    "contains": None,
                    "observed_count": item.get("transition_count"),
                    "evidence": [evidence],
                    "action": item.get("action_type") or "reduce_transition_gap",
                    "action_type": item.get("action_type") or "reduce_transition_gap",
                    "target": item.get("target") or transition,
                    "scenario_id": item.get("scenario_id"),
                    "delta_signature": item.get("delta_signature"),
                    "modeled_delta": float(delta_hours),
                    "measurement_type": item.get("measurement_type") or "modeled",
                    "impact_hours": impact_hours,
                    "confidence_weight": confidence_weight,
                    "controllability_weight": controllability_weight,
                    "relevance_score": relevance_score,
                    **context,
                }
            )

    user_host_plugin = plugins.get("analysis_user_host_savings")
    if isinstance(user_host_plugin, dict):
        findings = [
            f
            for f in user_host_plugin.get("findings") or []
            if isinstance(f, dict) and f.get("kind") == "user_host_savings"
        ]
        for item in findings:
            delta_hours = item.get("delta_hours") or item.get("delta_value")
            if not isinstance(delta_hours, (int, float)) or float(delta_hours) <= 0:
                continue
            dimension = item.get("dimension")
            group_id = item.get("group_id")
            context = _metric_context_from_item("user_host_savings", item, report)
            confidence_weight = item.get("confidence_weight")
            if not isinstance(confidence_weight, (int, float)):
                confidence_weight = _confidence_weight(item, {})
            controllability_weight = item.get("controllability_weight")
            if not isinstance(controllability_weight, (int, float)):
                controllability_weight = _controllability_weight(
                    "user_host_savings", {}
                )
            impact_hours = float(delta_hours)
            relevance_score = (
                impact_hours * float(confidence_weight) * float(controllability_weight)
            )
            evidence = {
                "dimension": dimension,
                "group_id": group_id,
                "baseline_hours": item.get("baseline_over_threshold_hours"),
                "modeled_hours": item.get("modeled_over_threshold_hours"),
                "delta_hours": delta_hours,
                "runs_count": item.get("runs_count"),
                "columns": item.get("columns"),
            }
            items.append(
                {
                    "title": f"Rebalance {dimension} {group_id}",
                    "status": "discovered",
                    "category": "discovery",
                    "recommendation": item.get("recommendation")
                    or f"Rebalance {dimension} {group_id} (Δ {float(delta_hours):.2f}h).",
                    "plugin_id": "analysis_user_host_savings",
                    "kind": "user_host_savings",
                    "where": {"dimension": dimension, "group_id": group_id},
                    "contains": None,
                    "observed_count": item.get("runs_count"),
                    "evidence": [evidence],
                    "action": item.get("action_type") or "rebalance_assignment",
                    "action_type": item.get("action_type") or "rebalance_assignment",
                    "target": item.get("target") or group_id,
                    "scenario_id": item.get("scenario_id"),
                    "delta_signature": item.get("delta_signature"),
                    "modeled_delta": float(delta_hours),
                    "measurement_type": item.get("measurement_type") or "modeled",
                    "impact_hours": impact_hours,
                    "confidence_weight": confidence_weight,
                    "controllability_weight": controllability_weight,
                    "relevance_score": relevance_score,
                    **context,
                }
            )

    linkage_plugin = plugins.get("analysis_upload_linkage")
    if isinstance(linkage_plugin, dict):
        findings = [
            f
            for f in linkage_plugin.get("findings") or []
            if isinstance(f, dict) and f.get("kind") == "upload_bkrvnu_linkage"
        ]
        for item in findings:
            matched_user_pct = item.get("matched_user_pct")
            matched_any_pct = item.get("matched_any_pct")
            bkrvnu_rows = item.get("bkrvnu_rows")
            upload_rows = item.get("upload_rows")
            window_hours = item.get("window_hours")
            upload_process = item.get("upload_process") or "upload"
            if not isinstance(matched_user_pct, (int, float)):
                continue
            if matched_user_pct >= 0.5:
                continue
            unmatched = None
            if isinstance(bkrvnu_rows, (int, float)) and isinstance(
                item.get("matched_user_count"), (int, float)
            ):
                unmatched = float(bkrvnu_rows) - float(item.get("matched_user_count"))
            context = _metric_context_from_item("upload_bkrvnu_linkage", item, report)
            rec_text = (
                f"Add explicit linkage from XLSX uploads to downstream revenue booking: "
                f"only {matched_user_pct*100:.1f}% of BKRVNU rows match an upload by user "
                f"+{int(window_hours) if isinstance(window_hours, (int, float)) else 'n/a'}h window."
            )
            if isinstance(matched_any_pct, (int, float)):
                rec_text += f" Even relaxed matching is only {matched_any_pct*100:.1f}%."
            rec_text += " Capture the upload batch/filename and propagate it into revenue booking metadata."
            confidence_weight = _confidence_weight(item, {})
            controllability_weight = _controllability_weight(
                "upload_bkrvnu_linkage", {}
            )
            impact_hours = float(unmatched) if unmatched is not None else float(bkrvnu_rows or 0.0)
            relevance_score = impact_hours * confidence_weight * controllability_weight
            evidence = {
                "upload_process": upload_process,
                "upload_rows": upload_rows,
                "bkrvnu_rows": bkrvnu_rows,
                "matched_user_pct": matched_user_pct,
                "matched_any_pct": matched_any_pct,
                "window_hours": window_hours,
            }
            items.append(
                {
                    "title": "Add deterministic XLSX-to-revenue linkage",
                    "status": "discovered",
                    "category": "discovery",
                    "recommendation": rec_text,
                    "plugin_id": "analysis_upload_linkage",
                    "kind": "upload_bkrvnu_linkage",
                    "where": {"process_norm": upload_process},
                    "contains": None,
                    "observed_count": bkrvnu_rows,
                    "evidence": [evidence],
                    "action": "add_upload_linkage",
                    "modeled_delta": None,
                    "measurement_type": item.get("measurement_type") or "measured",
                    "impact_hours": impact_hours,
                    "confidence_weight": confidence_weight,
                    "controllability_weight": controllability_weight,
                    "relevance_score": relevance_score,
                    **context,
                }
            )

    # Family F: CEO-grade operational levers (process-targeted actions).
    # Support levers emitted by any plugin (not just analysis_actionable_ops_levers_v1).
    ops_findings: list[tuple[str, dict[str, Any]]] = []
    for pid, plugin in plugins.items():
        if not isinstance(plugin, dict):
            continue
        for f in plugin.get("findings") or []:
            if isinstance(f, dict) and f.get("kind") == "actionable_ops_lever":
                ops_findings.append((pid, f))

    if ops_findings:
        for src_plugin_id, item in ops_findings:
            base_evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
            selected_rows = (
                base_evidence.get("selected") if isinstance(base_evidence.get("selected"), list) else []
            )
            expanded: list[dict[str, Any]] = []
            for row in selected_rows:
                if not isinstance(row, dict):
                    continue
                candidate = dict(item)
                for key in ("process_norm", "process", "process_id", "action_type", "title"):
                    if row.get(key):
                        candidate[key] = row.get(key)
                if isinstance(row.get("delta_seconds"), (int, float)):
                    candidate["expected_delta_seconds"] = float(row.get("delta_seconds") or 0.0)
                candidate["_selected_row"] = row
                expanded.append(candidate)
            if not expanded:
                targets = _target_process_ids_for_finding(item)
                if targets:
                    for process in targets:
                        candidate = dict(item)
                        candidate["process_norm"] = process
                        candidate["process"] = process
                        expanded.append(candidate)
                else:
                    expanded.append(dict(item))

            for candidate in expanded:
                process = _normalize_process_hint(
                    candidate.get("process_norm")
                    or candidate.get("process")
                    or candidate.get("process_id")
                )
                if not process:
                    continue
                action_type = str(candidate.get("action_type") or "").strip().lower()
                if not action_type:
                    action_type = _infer_action_type_from_text(
                        _finding_recommendation_text(candidate) or str(candidate.get("title") or "")
                    )
                if not action_type:
                    action_type = "actionable_ops"
                if action_type in {"unblock_dependency_chain", "orchestrate_chain"}:
                    # Environment policy: chain/flow rewiring is not modifiable.
                    continue
                title = candidate.get("title") or f"Operational lever for {process}"
                delta_s = candidate.get("expected_delta_seconds")
                delta_pct = candidate.get("expected_delta_percent")
                conf = candidate.get("confidence")
                if not isinstance(conf, (int, float)):
                    conf = _confidence_weight(candidate, {})
                impact_hours = float(delta_s) / 3600.0 if isinstance(delta_s, (int, float)) else 0.0
                controllability_weight = 0.9
                relevance_score = impact_hours * float(conf) * controllability_weight
                context = _metric_context_from_item("actionable_ops_lever", candidate, report)
                rec_text = _finding_recommendation_text(candidate) or _finding_recommendation_text(item)
                evidence = dict(base_evidence) if isinstance(base_evidence, dict) else {"source": src_plugin_id}
                selected_row = candidate.get("_selected_row")
                if isinstance(selected_row, dict):
                    evidence["selected_action"] = selected_row
                if excluded_match and excluded_match(process):
                    continue
                if not isinstance(rec_text, str) or not rec_text.strip():
                    rec_text = f"{title}."
                    if isinstance(delta_s, (int, float)):
                        rec_text += f" Expected Δ {float(delta_s):.2f}s"
                    if isinstance(delta_pct, (int, float)):
                        rec_text += f" ({float(delta_pct):.1f}%)."
                items.append(
                    {
                        "title": title,
                        "status": "discovered",
                        "category": "discovery",
                        "recommendation": rec_text,
                        "plugin_id": src_plugin_id,
                        "kind": "actionable_ops_lever",
                        "where": {"process_norm": process},
                        "contains": None,
                        "observed_count": None,
                        "evidence": [evidence],
                        "action": action_type,
                        "action_type": action_type,
                        "target": process,
                        "scenario_id": candidate.get("process_id"),
                        "delta_signature": None,
                        "modeled_delta": impact_hours if impact_hours else None,
                        "measurement_type": candidate.get("measurement_type") or "measured",
                        "impact_hours": impact_hours,
                        "confidence_weight": float(conf),
                        "controllability_weight": controllability_weight,
                        "relevance_score": relevance_score,
                        **context,
                    }
                )

    ideaspace_plugin = plugins.get("analysis_ideaspace_action_planner")
    if isinstance(ideaspace_plugin, dict):
        idea_findings = [
            f
            for f in (ideaspace_plugin.get("findings") or [])
            if isinstance(f, dict) and f.get("kind") == "ideaspace_action"
        ]
        for item in idea_findings:
            lever_id = str(item.get("lever_id") or "").strip().lower()
            text = str(item.get("what") or "").strip()
            if text.lower().startswith("action: "):
                text = text[8:].strip()
            if not text:
                recs = item.get("recommendations")
                if isinstance(recs, list) and recs:
                    text = str(recs[0]).strip()
            if not text:
                text = str(item.get("recommendation") or "").strip()
            if not text:
                continue

            action_type = "ideaspace_action"
            target = str(item.get("process_id") or "").strip() or None
            if lever_id == "tune_schedule_qemail_frequency_v1" or "qemail" in text.lower():
                action_type = "tune_schedule"
                target = "qemail"
            elif lever_id == "add_qpec_capacity_plus_one_v1" or "qpec" in text.lower():
                action_type = "add_server"
                target = "qpec"
            elif lever_id == "split_batches":
                action_type = "batch_input_refactor"
            elif lever_id == "priority_isolation":
                action_type = "orchestrate_macro"
            elif lever_id == "retry_backoff":
                action_type = "dedupe_or_cache"
            elif "instrumentation" in text.lower() or "trace" in text.lower():
                action_type = "add_instrumentation"

            if isinstance(target, str) and target and excluded_match and excluded_match(target):
                if action_type not in {"add_server", "tune_schedule"}:
                    continue

            confidence_weight = _confidence_weight(item, {})
            controllability_weight = 0.8 if action_type in {"add_server", "tune_schedule"} else 0.6
            impact_pct = item.get("delta_value")
            if not isinstance(impact_pct, (int, float)):
                impact_pct = item.get("estimated_delta_pct")
            impact_hours = 0.0
            est_hours_total = item.get("estimated_delta_hours_total")
            est_delta_seconds = item.get("estimated_delta_seconds")
            if isinstance(est_hours_total, (int, float)):
                impact_hours = max(0.0, float(est_hours_total))
            elif isinstance(est_delta_seconds, (int, float)):
                impact_hours = max(0.0, float(est_delta_seconds) / 3600.0)
            modeled_percent_hint: float | None = None
            unit = str(item.get("unit") or "").strip().lower()
            if isinstance(impact_pct, (int, float)):
                if unit in {"percent", "pct", "%"}:
                    raw_pct = float(impact_pct)
                    modeled_percent_hint = raw_pct * 100.0 if 0.0 <= raw_pct <= 1.0 else raw_pct
                else:
                    # Back-compat: when unit is unspecified but field name implies pct,
                    # treat values <=1 as ratio and >1 as already-percent.
                    raw_pct = float(impact_pct)
                    modeled_percent_hint = raw_pct * 100.0 if 0.0 <= raw_pct <= 1.0 else raw_pct
            if not isinstance(modeled_percent_hint, (int, float)) and impact_hours <= 0.0:
                if isinstance(item.get("estimated_delta_hours_total"), (int, float)):
                    impact_hours = max(0.0, float(item.get("estimated_delta_hours_total")))
                elif isinstance(item.get("estimated_delta_seconds"), (int, float)):
                    impact_hours = max(0.0, float(item.get("estimated_delta_seconds")) / 3600.0)
            relevance_basis = (
                float(modeled_percent_hint)
                if isinstance(modeled_percent_hint, (int, float))
                else float(impact_hours)
            )
            relevance_score = relevance_basis * float(confidence_weight) * float(controllability_weight)
            evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
            where = {"process_norm": target} if isinstance(target, str) and target else None

            items.append(
                {
                    "title": str(item.get("title") or "Ideaspace action").strip(),
                    "status": "discovery",
                    "category": "discovery",
                    "recommendation": text,
                    "plugin_id": "analysis_ideaspace_action_planner",
                    "kind": "ideaspace_action",
                    "where": where,
                    "process_id": target,
                    "contains": None,
                    "observed_count": None,
                    "evidence": [evidence] if evidence else [],
                    "action": action_type,
                    "action_type": action_type,
                    "target": target,
                    "scenario_id": item.get("id"),
                    "delta_signature": None,
                    "modeled_delta": impact_hours if impact_hours > 0.0 else None,
                    "modeled_percent_hint": modeled_percent_hint,
                    "unit": unit or None,
                    "measurement_type": item.get("measurement_type") or "modeled",
                    "impact_hours": impact_hours,
                    "estimated_delta_hours_total": (
                        float(est_hours_total) if isinstance(est_hours_total, (int, float)) else None
                    ),
                    "estimated_delta_seconds": (
                        float(est_delta_seconds) if isinstance(est_delta_seconds, (int, float)) else None
                    ),
                    "confidence_weight": float(confidence_weight),
                    "controllability_weight": float(controllability_weight),
                    "relevance_score": float(relevance_score),
                }
            )

    ideaspace_verified_plugin = plugins.get("analysis_ebm_action_verifier_v1")
    if isinstance(ideaspace_verified_plugin, dict):
        verified_findings = [
            f
            for f in (ideaspace_verified_plugin.get("findings") or [])
            if isinstance(f, dict) and f.get("kind") == "verified_action"
        ]
        for item in verified_findings:
            text = str(item.get("what") or "").strip()
            if not text:
                recs = item.get("recommendations")
                if isinstance(recs, list) and recs:
                    text = str(recs[0]).strip()
            if not text:
                continue

            lever_id = str(item.get("lever_id") or "").strip().lower()
            action_type = "ideaspace_action"
            target = str(item.get("target") or "").strip() or None
            if lever_id == "tune_schedule_qemail_frequency_v1" or ("qemail" in text.lower()):
                action_type = "tune_schedule"
                if not target:
                    target = "qemail"
            elif lever_id == "add_qpec_capacity_plus_one_v1" or ("qpec" in text.lower()):
                action_type = "add_server"
                if not target:
                    target = "qpec"
            elif lever_id == "split_batches":
                action_type = "batch_input_refactor"
            elif lever_id == "priority_isolation":
                action_type = "orchestrate_macro"
            elif lever_id == "retry_backoff":
                action_type = "dedupe_or_cache"
            elif lever_id == "cap_concurrency":
                action_type = "cap_concurrency"
            elif lever_id == "blackout_scheduled_jobs":
                action_type = "schedule_shift_target"

            if isinstance(target, str) and target and excluded_match and excluded_match(target):
                if action_type not in {"add_server", "tune_schedule"}:
                    continue

            confidence_weight = _confidence_weight(item, {})
            controllability_weight = 0.75 if action_type in {"add_server", "tune_schedule"} else 0.60
            delta_energy = item.get("delta_energy")
            energy_before = item.get("energy_before")
            modeled_percent_hint: float | None = None
            if isinstance(delta_energy, (int, float)) and isinstance(energy_before, (int, float)) and float(energy_before) > 0.0:
                modeled_percent_hint = max(0.0, min(100.0, (float(delta_energy) / float(energy_before)) * 100.0))
            relevance_basis = float(delta_energy) if isinstance(delta_energy, (int, float)) else (
                float(modeled_percent_hint) if isinstance(modeled_percent_hint, (int, float)) else 0.0
            )
            relevance_score = relevance_basis * float(confidence_weight) * float(controllability_weight)
            evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
            where = {"process_norm": target} if isinstance(target, str) and target and "," not in target else None

            items.append(
                {
                    "title": str(item.get("title") or "Verified Kona action").strip(),
                    "status": "discovered",
                    "category": "discovery",
                    "recommendation": text,
                    "plugin_id": "analysis_ebm_action_verifier_v1",
                    "kind": "verified_action",
                    "where": where,
                    "contains": None,
                    "observed_count": None,
                    "evidence": [evidence] if evidence else [],
                    "action": action_type,
                    "action_type": action_type,
                    "target": target,
                    "scenario_id": item.get("id"),
                    "delta_signature": None,
                    "modeled_delta": None,
                    "modeled_percent_hint": modeled_percent_hint,
                    "unit": "percent" if isinstance(modeled_percent_hint, (int, float)) else "energy_points",
                    "measurement_type": item.get("measurement_type") or "modeled",
                    "impact_hours": 0.0,
                    "confidence_weight": float(confidence_weight),
                    "controllability_weight": float(controllability_weight),
                    "relevance_score": float(relevance_score),
                    "delta_energy": float(delta_energy) if isinstance(delta_energy, (int, float)) else None,
                    "energy_before": float(energy_before) if isinstance(energy_before, (int, float)) else None,
                    "energy_after": float(item.get("energy_after")) if isinstance(item.get("energy_after"), (int, float)) else None,
                }
            )

    # Spillover past EOM: baseline close-window vs target close-window gap.
    for pid in ("analysis_close_cycle_capacity_model", "analysis_close_cycle_duration_shift"):
        plug = plugins.get(pid)
        if not isinstance(plug, dict):
            continue
        spill = [
            f
            for f in (plug.get("findings") or [])
            if isinstance(f, dict) and f.get("kind") == "spillover_past_eom"
        ]
        if not spill:
            continue
        item = spill[0]
        detail = item.get("details") if isinstance(item.get("details"), dict) else {}
        spill_evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
        top_proc = detail.get("top_spillover_processes")
        if not isinstance(top_proc, list):
            top_proc = spill_evidence.get("top_spillover_processes")
        proc_rows: list[dict[str, Any]] = []
        if isinstance(top_proc, list):
            proc_rows = [r for r in top_proc if isinstance(r, dict)]

        qh = item.get("spillover_queue_wait_hours_total")
        sh = item.get("spillover_service_hours_total")
        dh = item.get("spillover_duration_hours_total")
        rows_total = item.get("spillover_rows_total")
        parts = []
        if isinstance(rows_total, (int, float)):
            parts.append(f"{int(rows_total):,} runs")
        if isinstance(qh, (int, float)):
            parts.append(f"{float(qh):.2f}h queue-wait")
        if isinstance(sh, (int, float)):
            parts.append(f"{float(sh):.2f}h service")
        if isinstance(dh, (int, float)):
            parts.append(f"{float(dh):.2f}h duration")
        metric_txt = ", ".join(parts) if parts else "measured spillover"
        rec_text = (
            "Close-cycle spillover past month-end exists under the current baseline close window "
            "(e.g. 20th->5th) that is outside the target close window (20th->EOM/31). "
            f"Observed spillover: {metric_txt}. "
            "Action: reclaim this spillover before EOM by targeting the highest-hour contributors first."
        )
        if proc_rows:
            top_total = 0.0
            if isinstance(qh, (int, float)) and float(qh) > 0.0:
                for r in proc_rows[:3]:
                    v = r.get("spillover_queue_wait_hours_total")
                    if isinstance(v, (int, float)):
                        top_total += float(v)
                if top_total > 0.0:
                    rec_text += (
                        f" Top 3 contributors represent ~{(top_total / float(qh)) * 100.0:.1f}% "
                        "of total spillover queue-wait."
                    )
            proc_summ: list[str] = []
            for r in proc_rows[:8]:
                proc = r.get("process_norm") or r.get("process")
                hrs = r.get("spillover_queue_wait_hours_total") or r.get("spillover_duration_hours_total") or r.get("spillover_service_hours_total")
                if isinstance(proc, str) and proc.strip():
                    proc_norm = proc.strip().lower()
                    if isinstance(hrs, (int, float)):
                        proc_summ.append(f"{proc_norm} (~{float(hrs):.1f}h)")
                    else:
                        proc_summ.append(proc_norm)
            if proc_summ:
                rec_text += " Top spillover processes: " + ", ".join(proc_summ) + "."

        artifacts_paths = _artifact_paths(report, pid)
        spill_path = next((p for p in artifacts_paths if p.endswith("spillover_target_window.json")), None)
        evidence = {
            "source_plugin": pid,
            "spillover_artifact": spill_path,
            "spillover_rows_total": rows_total,
            "spillover_queue_wait_hours_total": qh,
            "spillover_service_hours_total": sh,
            "spillover_duration_hours_total": dh,
        }
        impact_hours = float(qh) if isinstance(qh, (int, float)) else (float(dh) if isinstance(dh, (int, float)) else 0.0)
        items.append(
            {
                "title": "Reduce close-cycle spillover past EOM",
                "status": "discovered",
                "category": "discovery",
                "recommendation": rec_text,
                "plugin_id": pid,
                "kind": "spillover_past_eom",
                "where": None,
                "contains": None,
                "observed_count": rows_total,
                "evidence": [evidence],
                "action": "reduce_spillover_past_eom",
                "action_type": "reduce_spillover_past_eom",
                "modeled_delta": None,
                "measurement_type": item.get("measurement_type") or "measured",
                "impact_hours": impact_hours,
                "confidence_weight": 0.7,
                "controllability_weight": 0.8,
                "relevance_score": impact_hours * 0.7 * 0.8,
            }
        )

    close_capacity_model = plugins.get("analysis_close_cycle_capacity_model")
    if isinstance(close_capacity_model, dict):
        for item in close_capacity_model.get("findings") or []:
            if not isinstance(item, dict) or item.get("kind") != "close_cycle_capacity_model":
                continue
            if str(item.get("decision") or "").strip().lower() != "modeled":
                continue
            modeled_percent_hint = _modeled_improvement_percent(
                item.get("baseline_value"), item.get("modeled_value")
            )
            if not isinstance(modeled_percent_hint, (int, float)) or float(modeled_percent_hint) <= 0.0:
                continue
            baseline_value = item.get("baseline_value")
            modeled_value = item.get("modeled_value")
            delta_seconds = (
                float(baseline_value) - float(modeled_value)
                if isinstance(baseline_value, (int, float)) and isinstance(modeled_value, (int, float))
                else 0.0
            )
            impact_hours = max(0.0, delta_seconds / 3600.0)
            confidence_weight = _confidence_weight(item, {})
            controllability_weight = _controllability_weight("close_cycle_capacity_model", {})
            relevance_basis = float(modeled_percent_hint) if float(modeled_percent_hint) > 0.0 else impact_hours
            relevance_score = relevance_basis * float(confidence_weight) * float(controllability_weight)
            host_metric = str(item.get("host_metric") or "host_count").strip() or "host_count"
            metric_type = str(item.get("metric_type") or "close_cycle_duration").strip() or "close_cycle_duration"
            context = _metric_context_from_item("close_cycle_capacity_model", item, report)
            rec_text = (
                f"Add one server for close-cycle {host_metric}/{metric_type}: modeled median improves "
                f"from {float(baseline_value):.2f}s to {float(modeled_value):.2f}s "
                f"(~{float(modeled_percent_hint):.1f}% reduction)."
            )
            items.append(
                {
                    "title": f"Add close-cycle capacity ({host_metric}/{metric_type})",
                    "status": "discovered",
                    "category": "discovery",
                    "recommendation": rec_text,
                    "plugin_id": "analysis_close_cycle_capacity_model",
                    "kind": "close_cycle_capacity_model",
                    "where": None,
                    "contains": None,
                    "observed_count": item.get("bucket_count"),
                    "evidence": [
                        {
                            "host_metric": host_metric,
                            "metric_type": metric_type,
                            "baseline_value": baseline_value,
                            "modeled_value": modeled_value,
                            "scale_factor": item.get("scale_factor"),
                            "target_reduction": item.get("target_reduction"),
                        }
                    ],
                    "action": "add_server",
                    "action_type": "add_server",
                    "modeled_delta": impact_hours if impact_hours > 0.0 else None,
                    "modeled_percent_hint": float(modeled_percent_hint),
                    "unit": "percent",
                    "measurement_type": item.get("measurement_type") or "modeled",
                    "impact_hours": impact_hours,
                    "confidence_weight": float(confidence_weight),
                    "controllability_weight": float(controllability_weight),
                    "relevance_score": float(relevance_score),
                    **context,
                }
            )

    close_capacity_impact = plugins.get("analysis_close_cycle_capacity_impact")
    if isinstance(close_capacity_impact, dict):
        for item in close_capacity_impact.get("findings") or []:
            if not isinstance(item, dict) or item.get("kind") != "close_cycle_capacity_impact":
                continue
            if str(item.get("decision") or "").strip().lower() != "detected":
                continue
            effect = item.get("effect")
            if not isinstance(effect, (int, float)) or float(effect) >= 0.0:
                continue
            modeled_percent_hint = abs(float(effect)) * 100.0
            confidence_weight = _confidence_weight(item, {})
            controllability_weight = _controllability_weight("close_cycle_capacity_impact", {})
            relevance_score = (
                float(modeled_percent_hint)
                * float(confidence_weight)
                * float(controllability_weight)
            )
            host_metric = str(item.get("host_metric") or "host_count").strip() or "host_count"
            rec_text = (
                f"Add one server: measured close-cycle {host_metric} improves by "
                f"~{float(modeled_percent_hint):.1f}% in high-capacity buckets."
            )
            context = _metric_context_from_item("close_cycle_capacity_impact", item, report)
            items.append(
                {
                    "title": f"Add measured close-cycle capacity ({host_metric})",
                    "status": "discovered",
                    "category": "discovery",
                    "recommendation": rec_text,
                    "plugin_id": "analysis_close_cycle_capacity_impact",
                    "kind": "close_cycle_capacity_impact",
                    "where": None,
                    "contains": None,
                    "observed_count": item.get("hi_bucket_count"),
                    "evidence": [
                        {
                            "host_metric": host_metric,
                            "effect": effect,
                            "ci_low": item.get("ci_low"),
                            "ci_high": item.get("ci_high"),
                        }
                    ],
                    "action": "add_server",
                    "action_type": "add_server",
                    "modeled_delta": None,
                    "modeled_percent_hint": float(modeled_percent_hint),
                    "unit": "percent",
                    "measurement_type": item.get("measurement_type") or "measured",
                    "impact_hours": 0.0,
                    "confidence_weight": float(confidence_weight),
                    "controllability_weight": float(controllability_weight),
                    "relevance_score": float(relevance_score),
                    **context,
                }
            )

    close_revenue = plugins.get("analysis_close_cycle_revenue_compression")
    if isinstance(close_revenue, dict):
        for item in close_revenue.get("findings") or []:
            if not isinstance(item, dict) or item.get("kind") != "close_cycle_revenue_compression":
                continue
            if str(item.get("decision") or "").strip().lower() != "modeled":
                continue
            modeled_percent_hint = _modeled_improvement_percent(
                item.get("baseline_value"), item.get("modeled_value")
            )
            if not isinstance(modeled_percent_hint, (int, float)) or float(modeled_percent_hint) <= 0.0:
                continue
            process_norm = str(item.get("process_label") or "").strip().lower() or None
            if process_norm and excluded_match and excluded_match(process_norm):
                continue
            baseline_days = item.get("baseline_value")
            modeled_days = item.get("modeled_value")
            impact_hours = (
                max(0.0, float(baseline_days) - float(modeled_days)) * 24.0
                if isinstance(baseline_days, (int, float)) and isinstance(modeled_days, (int, float))
                else 0.0
            )
            confidence_weight = _confidence_weight(item, {})
            controllability_weight = _controllability_weight("close_cycle_revenue_compression", {})
            relevance_basis = float(modeled_percent_hint) if float(modeled_percent_hint) > 0.0 else impact_hours
            relevance_score = relevance_basis * float(confidence_weight) * float(controllability_weight)
            process_label = str(item.get("process_label") or process_norm or "revenue_close").strip()
            rec_text = (
                f"Add revenue-close capacity for {process_label}: modeled span improves from "
                f"{float(baseline_days):.2f} days to {float(modeled_days):.2f} days "
                f"(~{float(modeled_percent_hint):.1f}% reduction)."
            )
            context = _metric_context_from_item("close_cycle_revenue_compression", item, report)
            items.append(
                {
                    "title": f"Add revenue-close capacity ({process_label})",
                    "status": "discovered",
                    "category": "discovery",
                    "recommendation": rec_text,
                    "plugin_id": "analysis_close_cycle_revenue_compression",
                    "kind": "close_cycle_revenue_compression",
                    "where": {"process_norm": process_norm} if process_norm else None,
                    "contains": None,
                    "observed_count": None,
                    "evidence": [
                        {
                            "process_label": process_label,
                            "baseline_value_days": baseline_days,
                            "modeled_value_days": modeled_days,
                            "required_scale_factor_median": item.get("required_scale_factor_median"),
                        }
                    ],
                    "action": "add_server",
                    "action_type": "add_server",
                    "modeled_delta": impact_hours if impact_hours > 0.0 else None,
                    "modeled_percent_hint": float(modeled_percent_hint),
                    "unit": "percent",
                    "measurement_type": item.get("measurement_type") or "modeled",
                    "impact_hours": impact_hours,
                    "confidence_weight": float(confidence_weight),
                    "controllability_weight": float(controllability_weight),
                    "relevance_score": float(relevance_score),
                    **context,
                }
            )

    close_uplift = plugins.get("analysis_close_cycle_uplift")
    if isinstance(close_uplift, dict):
        for item in close_uplift.get("findings") or []:
            if not isinstance(item, dict) or item.get("kind") != "close_cycle_share_shift":
                continue
            process_norm = str(item.get("process_norm") or item.get("process") or "").strip().lower()
            if not process_norm:
                continue
            if excluded_match and excluded_match(process_norm):
                continue
            share_delta = item.get("share_delta")
            if not isinstance(share_delta, (int, float)) or float(share_delta) <= 0.0:
                continue
            share_delta_pp = float(share_delta) * 100.0
            close_count = item.get("close_count")
            median_close = item.get("median_close")
            median_open = item.get("median_open")
            impact_hours = 0.0
            if (
                isinstance(close_count, (int, float))
                and isinstance(median_close, (int, float))
                and isinstance(median_open, (int, float))
                and float(median_close) > float(median_open)
            ):
                impact_hours = (
                    max(0.0, float(median_close) - float(median_open))
                    * max(0.0, float(close_count))
                    / 3600.0
                )
            confidence_weight = _confidence_weight(item, {})
            controllability_weight = 0.75
            relevance_basis = share_delta_pp if share_delta_pp > 0.0 else impact_hours
            relevance_score = relevance_basis * float(confidence_weight) * float(controllability_weight)
            slowdown_ratio = item.get("slowdown_ratio")
            process_label = str(item.get("process") or process_norm).strip()
            rec_text = (
                f"Reduce close-cycle concentration for {process_label}: this process is over-indexed in close by "
                f"{share_delta_pp:.2f} percentage points"
            )
            if isinstance(slowdown_ratio, (int, float)) and float(slowdown_ratio) > 0.0:
                rec_text += f" with close-cycle slowdown ratio {float(slowdown_ratio):.2f}x."
            else:
                rec_text += "."
            rec_text += (
                " Action: split oversized inputs and move pre-close steps earlier so this process does less work inside the close window."
            )
            context = _metric_context_from_item("close_cycle_duration_shift", item, report)
            items.append(
                {
                    "title": f"Reduce close-cycle concentration ({process_label})",
                    "status": "discovered",
                    "category": "discovery",
                    "recommendation": rec_text,
                    "plugin_id": "analysis_close_cycle_uplift",
                    "kind": "close_cycle_share_shift",
                    "where": {"process_norm": process_norm},
                    "contains": None,
                    "observed_count": close_count if isinstance(close_count, (int, float)) else None,
                    "evidence": [
                        {
                            "process_norm": process_norm,
                            "close_share": item.get("close_share"),
                            "open_share": item.get("open_share"),
                            "share_delta": share_delta,
                            "slowdown_ratio": slowdown_ratio,
                        }
                    ],
                    "action": "reduce_transition_gap",
                    "action_type": "reduce_transition_gap",
                    "target": process_norm,
                    "modeled_delta": impact_hours if impact_hours > 0.0 else None,
                    "modeled_percent_hint": share_delta_pp,
                    "unit": "percent",
                    "measurement_type": item.get("measurement_type") or "measured",
                    "impact_hours": impact_hours,
                    "confidence_weight": float(confidence_weight),
                    "controllability_weight": float(controllability_weight),
                    "relevance_score": float(relevance_score),
                    **context,
                }
            )

    # Sort before dedupe so that higher-value items "win" when text merges occur.
    items = sorted(items, key=_discovery_recommendation_sort_key, reverse=True)

    controls = _recommendation_controls(report)
    suppressed = _suppressed_action_types()
    if not os.environ.get(_SUPPRESS_ACTION_TYPES_ENV, "").strip():
        extra_suppressed = controls.get("suppress_action_types")
        if isinstance(extra_suppressed, list):
            for token in extra_suppressed:
                if isinstance(token, str) and token.strip():
                    suppressed.add(token.strip())

    caps = _max_per_action_type()
    if not os.environ.get(_MAX_PER_ACTION_TYPE_ENV, "").strip():
        extra_caps = controls.get("max_per_action_type")
        if isinstance(extra_caps, dict):
            for key, value in extra_caps.items():
                if not isinstance(key, str):
                    continue
                try:
                    parsed = int(value)
                except (TypeError, ValueError):
                    continue
                if parsed > 0:
                    caps[key.strip()] = parsed

    allowed_action_types = _allow_action_types()
    if not allowed_action_types:
        extra_allowed = controls.get("allow_action_types")
        if isinstance(extra_allowed, list):
            allowed_action_types = {
                token.strip()
                for token in extra_allowed
                if isinstance(token, str) and token.strip()
            }

    allowed_process_patterns = _allow_process_patterns()
    if not allowed_process_patterns:
        extra_allow_processes = controls.get("allow_processes")
        if isinstance(extra_allow_processes, list):
            allowed_process_patterns = [
                token.strip()
                for token in extra_allow_processes
                if isinstance(token, str) and token.strip()
            ]
    allow_process_match = (
        compile_patterns(sorted(set(allowed_process_patterns)))
        if allowed_process_patterns
        else None
    )

    min_relevance = _min_relevance_score()
    if min_relevance <= 0.0 and isinstance(controls.get("min_relevance_score"), (int, float)):
        min_relevance = float(controls.get("min_relevance_score") or 0.0)
        if min_relevance < 0.0:
            min_relevance = 0.0

    top_n = _discovery_top_n()
    if top_n is None and isinstance(controls.get("top_n"), (int, float)):
        try:
            parsed_top_n = int(controls.get("top_n"))
        except (TypeError, ValueError):
            parsed_top_n = 0
        if parsed_top_n > 0:
            top_n = parsed_top_n

    max_obviousness = _max_obviousness()
    if isinstance(controls.get("max_obviousness"), (int, float)):
        if not os.environ.get(_MAX_OBVIOUSNESS_ENV, "").strip():
            ctrl_obv = float(controls.get("max_obviousness") or 0.0)
            if ctrl_obv < 0.0:
                ctrl_obv = 0.0
            if ctrl_obv > 1.0:
                ctrl_obv = 1.0
            max_obviousness = ctrl_obv

    kept: list[dict[str, Any]] = []
    used_by_action: dict[str, int] = {}
    for item in items:
        action_type = str(item.get("action_type") or item.get("action") or "").strip()
        obviousness_score = _action_type_obviousness(action_type)
        item["obviousness_score"] = float(obviousness_score)
        item["obviousness_rank"] = _obviousness_rank(float(obviousness_score))
        if allowed_action_types and action_type not in allowed_action_types:
            continue
        if action_type and action_type in suppressed:
            continue
        if float(obviousness_score) > float(max_obviousness):
            continue
        if allow_process_match is not None:
            process_hint = _recommendation_process_hint(item)
            if process_hint and not allow_process_match(process_hint):
                continue
        relevance = item.get("relevance_score")
        if not isinstance(relevance, (int, float)):
            relevance = item.get("impact_hours")
        if not isinstance(relevance, (int, float)):
            relevance = item.get("modeled_delta")
        if isinstance(relevance, (int, float)) and float(relevance) < min_relevance:
            continue
        if action_type:
            limit = caps.get(action_type)
            if isinstance(limit, int) and limit > 0:
                used_by_action[action_type] = int(used_by_action.get(action_type, 0)) + 1
                if used_by_action[action_type] > limit:
                    continue
        kept.append(item)

    deduped = _dedupe_recommendations_text(kept)
    baselines = _duration_baselines(report, storage, run_dir=run_dir)
    qemail_model = _process_removal_model(report, storage, "qemail", run_dir=run_dir)
    deduped = [
        _enrich_recommendation_item(item, report, baselines, qemail_model)
        for item in deduped
        if isinstance(item, dict)
    ]
    deduped = _apply_recency_weight(deduped)
    dropped_non_direct_process = 0
    dropped_chain_bound_process = 0
    if _require_direct_process_action():
        allowed_direct_actions = _direct_process_action_types()
        direct_process_hints = sorted(
            {
                str(_recommendation_process_hint(row) or "").strip().lower()
                for row in deduped
                if isinstance(row, dict)
            }
            - {""}
        )
        process_modifiability = _process_chain_modifiability_map(
            report, storage, direct_process_hints
        )
        direct_only: list[dict[str, Any]] = []
        for row in deduped:
            action_norm = str(row.get("action_type") or row.get("action") or "").strip().lower()
            if action_norm not in allowed_direct_actions:
                dropped_non_direct_process += 1
                continue
            process_hint = str(_recommendation_process_hint(row) or "").strip().lower()
            if not process_hint or process_hint in {"(multiple)", "all", "any", "global"}:
                dropped_non_direct_process += 1
                continue
            if process_hint in NON_ADJUSTABLE_PROCESSES:
                dropped_non_direct_process += 1
                continue
            if excluded_match and excluded_match(process_hint):
                dropped_non_direct_process += 1
                continue
            profile = process_modifiability.get(process_hint)
            if isinstance(profile, dict):
                row["lineage_modifiability"] = profile
                if bool(profile.get("chain_bound")):
                    dropped_chain_bound_process += 1
                    continue
            where = row.get("where") if isinstance(row.get("where"), dict) else {}
            if str(where.get("process_norm") or "").strip().lower() != process_hint:
                row["where"] = {"process_norm": process_hint}
            direct_only.append(row)
        deduped = direct_only
    dropped_without_modeled_hours = 0
    if _require_modeled_hours():
        normalized: list[dict[str, Any]] = []
        for row in deduped:
            modeled_hours = row.get("modeled_delta_hours")
            if isinstance(modeled_hours, (int, float)) and float(modeled_hours) > 0.0:
                row["relevance_score"] = float(modeled_hours)
                row["optimization_metric"] = "modeled_user_hours_saved"
                normalized.append(row)
                continue
            dropped_without_modeled_hours += 1
        deduped = normalized

    if _ranking_version() == "v2":
        for row in deduped:
            value_score = _safe_num(row.get("value_score_v2"))
            if not isinstance(value_score, (int, float)):
                value_score = _safe_num(row.get("client_value_score"))
            if isinstance(value_score, (int, float)):
                row["relevance_score"] = float(value_score)
                row["ranking_metric"] = "client_value_score_v2"

    deduped = sorted(deduped, key=_final_recommendation_sort_key)
    if isinstance(top_n, int) and top_n > 0:
        deduped = deduped[:top_n]
    status = "ok" if deduped else "none"
    summary = (
        f"Generated {len(deduped)} discovery recommendation(s) from plugin findings."
        if deduped
        else "No discovery recommendations generated from plugin findings."
    )
    if dropped_without_modeled_hours > 0:
        summary += (
            f" Dropped {dropped_without_modeled_hours} candidate(s) without positive "
            "modeled user-hours savings."
        )
    if dropped_non_direct_process > 0:
        summary += (
            f" Dropped {dropped_non_direct_process} candidate(s) that are not direct "
            "process-level actionable changes."
        )
    if dropped_chain_bound_process > 0:
        summary += (
            f" Dropped {dropped_chain_bound_process} candidate(s) because the target "
            "process is parent/child chain-bound (non-modifiable flow)."
        )
    return {"status": status, "summary": summary, "items": deduped}


def _manifest_index() -> dict[str, dict[str, Any]]:
    plugins_root = Path(__file__).resolve().parents[3] / "plugins"
    index: dict[str, dict[str, Any]] = {}
    for manifest in sorted(plugins_root.glob("*/plugin.yaml")):
        try:
            payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        plugin_id = str(payload.get("id") or manifest.parent.name)
        deps = payload.get("depends_on")
        depends_on = [str(v).strip() for v in deps] if isinstance(deps, list) else []
        index[plugin_id] = {
            "type": str(payload.get("type") or "").strip().lower(),
            "depends_on": [v for v in depends_on if v],
        }
    return index


def _plugin_class_taxonomy() -> dict[str, Any]:
    global _PLUGIN_CLASS_TAXONOMY_CACHE
    if _PLUGIN_CLASS_TAXONOMY_CACHE is not None:
        return _PLUGIN_CLASS_TAXONOMY_CACHE
    try:
        payload = yaml.safe_load(_PLUGIN_CLASS_TAXONOMY_PATH.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    _PLUGIN_CLASS_TAXONOMY_CACHE = payload if isinstance(payload, dict) else {}
    return _PLUGIN_CLASS_TAXONOMY_CACHE


def _plugin_class_id(plugin_id: str, plugin_type: str) -> str:
    taxonomy = _plugin_class_taxonomy()
    overrides = taxonomy.get("plugin_overrides") if isinstance(taxonomy.get("plugin_overrides"), dict) else {}
    defaults = (
        taxonomy.get("plugin_type_default_class")
        if isinstance(taxonomy.get("plugin_type_default_class"), dict)
        else {}
    )
    pid = str(plugin_id or "").strip()
    ptype = str(plugin_type or "").strip().lower()
    value = overrides.get(pid)
    if isinstance(value, str) and value.strip():
        return value.strip()
    default_value = defaults.get(ptype)
    if isinstance(default_value, str) and default_value.strip():
        return default_value.strip()
    if ptype and ptype != "analysis":
        return "supporting_signal_detectors"
    return "direct_action_generators" if pid == "analysis_actionable_ops_levers_v1" else "supporting_signal_detectors"


def _plugin_expected_output_type(plugin_class: str) -> str:
    taxonomy = _plugin_class_taxonomy()
    classes = taxonomy.get("classes") if isinstance(taxonomy.get("classes"), dict) else {}
    class_meta = classes.get(plugin_class) if isinstance(classes.get(plugin_class), dict) else {}
    value = class_meta.get("expected_output_type")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def _extract_precondition_inputs(payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else []
    debug = payload.get("debug") if isinstance(payload.get("debug"), dict) else {}
    required: set[str] = set()
    missing: set[str] = set()

    def _collect(target: set[str], value: Any) -> None:
        if isinstance(value, str):
            for token in re.split(r"[;,|]", value):
                item = token.strip()
                if item:
                    target.add(item)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                if isinstance(item, str) and item.strip():
                    target.add(item.strip())

    sources: list[dict[str, Any]] = [debug]
    sources.extend(item for item in findings if isinstance(item, dict))
    for item in sources:
        _collect(required, item.get("required_inputs"))
        _collect(required, item.get("required_columns"))
        _collect(required, item.get("required_fields"))
        _collect(missing, item.get("missing_inputs"))
        _collect(missing, item.get("missing_columns"))
        _collect(missing, item.get("missing_fields"))
        prereq = item.get("prerequisites") if isinstance(item.get("prerequisites"), dict) else {}
        if isinstance(prereq, dict):
            _collect(required, prereq.get("required"))
            _collect(required, prereq.get("required_inputs"))
            _collect(required, prereq.get("required_columns"))
            _collect(missing, prereq.get("missing"))
            _collect(missing, prereq.get("missing_inputs"))
            _collect(missing, prereq.get("missing_columns"))

    required_sorted = sorted(required)
    missing_sorted = sorted(missing)
    return required_sorted, missing_sorted


def _downstream_consumers(plugin_ids: set[str], manifest_index: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    reverse: dict[str, set[str]] = {pid: set() for pid in plugin_ids}
    for pid in plugin_ids:
        meta = manifest_index.get(pid) or {}
        deps = meta.get("depends_on") if isinstance(meta.get("depends_on"), list) else []
        for dep in deps:
            dep_id = str(dep or "").strip()
            if dep_id and dep_id in reverse:
                reverse[dep_id].add(pid)
    out: dict[str, list[str]] = {}
    for pid in sorted(plugin_ids):
        seen: set[str] = set()
        queue: deque[str] = deque(sorted(reverse.get(pid) or []))
        while queue:
            cur = queue.popleft()
            if cur in seen:
                continue
            seen.add(cur)
            for nxt in sorted(reverse.get(cur) or []):
                if nxt not in seen:
                    queue.append(nxt)
        out[pid] = sorted(seen)
    return out


def _actionable_plugin_ids(items: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("plugin_id") or "").strip()
        if pid:
            out.add(pid)
    return out


def _reason_code_for_non_actionable(
    status: str, finding_count: int, blank_kind_count: int, payload: dict[str, Any]
) -> str:
    debug = payload.get("debug") if isinstance(payload.get("debug"), dict) else {}
    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else []
    typed_findings = [f for f in findings if isinstance(f, dict)]
    finding_kinds = {str(f.get("kind") or "").strip() for f in typed_findings}
    reason = derive_reason_code(
        status=status,
        finding_count=int(finding_count or 0),
        blank_kind_count=int(blank_kind_count or 0),
        debug=debug if isinstance(debug, dict) else {},
        findings=typed_findings,
    )
    if reason == "NO_DECISION_SIGNAL":
        if "plugin_not_applicable" in finding_kinds:
            return "PLUGIN_PRECONDITION_UNMET"
        if "plugin_observation" in finding_kinds:
            return "OBSERVATION_ONLY"
        if "analysis_no_action_diagnostic" in finding_kinds:
            explicit = {
                str(f.get("reason_code") or "").strip()
                for f in typed_findings
                if str(f.get("kind") or "").strip() == "analysis_no_action_diagnostic"
                and str(f.get("reason_code") or "").strip()
            }
            if "NO_ELIGIBLE_SLICE" in explicit:
                return "NO_ELIGIBLE_SLICE"
            if "NO_SIGNIFICANT_EFFECT" in explicit:
                return "NO_SIGNIFICANT_EFFECT"
            return "NO_STATISTICAL_SIGNAL"
        return "NO_ACTIONABLE_FINDING_CLASS"
    if reason != "ADAPTER_RULE_MISSING":
        return reason

    recommended_rows = [f for f in typed_findings if _finding_recommendation_text(f)]
    if not recommended_rows:
        return "NO_ACTIONABLE_FINDING_CLASS"
    if not _require_direct_process_action():
        return reason

    allowed_actions = _direct_process_action_types()
    blocked_no_process_target = False
    blocked_action_type = False
    blocked_process_policy = False
    for finding in recommended_rows:
        process_targets = _target_process_ids_for_finding(finding)
        if not process_targets:
            blocked_no_process_target = True
            continue
        if all(proc in NON_ADJUSTABLE_PROCESSES for proc in process_targets):
            blocked_process_policy = True
            continue
        process_hint = next(
            (proc for proc in process_targets if proc not in NON_ADJUSTABLE_PROCESSES),
            "",
        )
        action_hint = str(finding.get("action_type") or finding.get("action") or "").strip().lower()
        if action_hint and action_hint not in allowed_actions:
            blocked_action_type = True
            continue
        return reason
    if blocked_no_process_target:
        return "NO_DIRECT_PROCESS_TARGET"
    if blocked_process_policy:
        return "EXCLUDED_BY_PROCESS_POLICY"
    if blocked_action_type:
        return "ACTION_TYPE_POLICY_BLOCK"
    return reason


def _build_non_actionable_explanations(
    report: dict[str, Any], recommendation_items: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    plugins = report.get("plugins") if isinstance(report.get("plugins"), dict) else {}
    plugin_ids = {str(pid) for pid in plugins.keys()}
    actionable_ids = _actionable_plugin_ids(recommendation_items)
    manifest_index = _manifest_index()
    downstream_map = _downstream_consumers(plugin_ids, manifest_index)
    items: list[dict[str, Any]] = []
    explained_ids: set[str] = set()
    for pid in sorted(plugin_ids):
        if pid in actionable_ids:
            continue
        payload = plugins.get(pid) if isinstance(plugins.get(pid), dict) else {}
        plugin_type = str((manifest_index.get(pid) or {}).get("type") or "unknown").strip().lower()
        plugin_class = _plugin_class_id(pid, plugin_type)
        expected_output_type = _plugin_expected_output_type(plugin_class)
        status = str(payload.get("status") or "unknown").strip().lower()
        summary = str(payload.get("summary") or "").strip()
        findings = payload.get("findings") if isinstance(payload.get("findings"), list) else []
        finding_count = int(len([f for f in findings if isinstance(f, dict)]))
        blank_kind_count = int(
            sum(
                1
                for f in findings
                if isinstance(f, dict) and not str(f.get("kind") or "").strip()
            )
        )
        top_kinds = Counter(
            str(f.get("kind") or "").strip()
            for f in findings
            if isinstance(f, dict) and str(f.get("kind") or "").strip()
        ).most_common(8)
        kind_preview = [k for k, _ in top_kinds]
        base_reason_code = _reason_code_for_non_actionable(
            status, finding_count, blank_kind_count, payload
        )
        reason_code = base_reason_code
        downstream = downstream_map.get(pid) or []
        required_inputs, missing_inputs = _extract_precondition_inputs(payload)
        explanation = plain_english_explanation(
            plugin_id=pid,
            plugin_type=plugin_type,
            status=status,
            summary=summary,
            finding_count=int(finding_count),
            blank_kind_count=int(blank_kind_count),
            downstream_plugins=downstream,
        )
        next_step = recommended_next_step(
            plugin_type=plugin_type,
            status=status,
            finding_count=int(finding_count),
            blank_kind_count=int(blank_kind_count),
            downstream_plugins=downstream,
        )
        if reason_code == "NO_DIRECT_PROCESS_TARGET":
            next_step = (
                "Emit process-level targets (`process_norm` or `target_process_ids`) for each action "
                "candidate, then rerun full gauntlet."
            )
        if reason_code == "NO_ACTIONABLE_FINDING_CLASS":
            next_step = (
                "Emit a direct-action finding kind (`actionable_ops_lever`, `ideaspace_action`, "
                "or `verified_action`) with process target and modeled delta fields."
            )
        if reason_code == "EXCLUDED_BY_PROCESS_POLICY":
            next_step = (
                "Current target is policy-blocked (chain-bound/non-adjustable). Promote the nearest "
                "modifiable parent process target and rerun."
            )
        if reason_code == "CAPACITY_IMPACT_CONSTRAINT":
            next_step = (
                "Capacity impact was not applicable for current slices; expand eligible slice coverage "
                "or provide modeled scenario inputs for capacity simulation."
            )
        if reason_code in {"PLUGIN_PRECONDITION_UNMET", "PREREQUISITE_UNMET"} and missing_inputs:
            next_step = (
                "Backfill missing inputs/columns and rerun full gauntlet: "
                + ", ".join(missing_inputs[:12])
                + ("." if len(missing_inputs) <= 12 else ", ...")
            )

        items.append(
            {
                "status": "explained_non_actionable",
                "plugin_id": pid,
                "plugin_type": plugin_type or "unknown",
                "plugin_status": status or "unknown",
                "kind": "non_actionable_explanation",
                "reason_code": reason_code,
                "reason_code_detail": base_reason_code if reason_code != base_reason_code else None,
                "plain_english_explanation": explanation,
                "recommended_next_step": next_step,
                "plugin_class": plugin_class,
                "expected_output_type": expected_output_type or None,
                "downstream_plugins": downstream,
                "downstream_plugin_count": int(len(downstream)),
                "finding_count": int(finding_count),
                "finding_kind_preview": kind_preview,
                "required_inputs": required_inputs,
                "missing_inputs": missing_inputs,
                "summary": summary,
            }
        )
        explained_ids.add(pid)

    lane_status = "ok" if items else "none"
    lane_summary = (
        f"Generated plain-English non-actionable explanations for {len(items)} plugin(s)."
        if items
        else "No non-actionable explanation entries were needed."
    )
    coverage = {
        "total_plugins": int(len(plugin_ids)),
        "actionable_plugin_count": int(len(actionable_ids)),
        "explained_non_actionable_count": int(len(explained_ids)),
        "unexplained_plugin_count": int(len(plugin_ids - actionable_ids - explained_ids)),
        "unexplained_plugins": sorted(plugin_ids - actionable_ids - explained_ids),
    }
    return {"status": lane_status, "summary": lane_summary, "items": items}, coverage


def _build_recommendations(
    report: dict[str, Any], storage: Storage | None = None, run_dir: Path | None = None
) -> dict[str, Any]:
    discovery = _build_discovery_recommendations(report, storage, run_dir=run_dir)
    known_payload = report.get("known_issues") if isinstance(report.get("known_issues"), dict) else None

    def _known_fingerprints() -> set[tuple[str, str, str | None]]:
        fps: set[tuple[str, str, str | None]] = set()
        if not isinstance(known_payload, dict):
            return fps
        expected = known_payload.get("expected_findings") or []
        if not isinstance(expected, list):
            return fps
        for exp in expected:
            if not isinstance(exp, dict):
                continue
            plugin_id = exp.get("plugin_id")
            kind = exp.get("kind")
            if not isinstance(plugin_id, str) or not isinstance(kind, str):
                continue
            proc = _process_hint(exp.get("where") if isinstance(exp.get("where"), dict) else None) or _process_hint(
                exp.get("contains") if isinstance(exp.get("contains"), dict) else None
            )
            proc_norm = proc.strip().lower() if isinstance(proc, str) and proc.strip() else None
            fps.add((plugin_id, kind, proc_norm))
        return fps

    def _matches_known(item: dict[str, Any], fp: tuple[str, str, str | None]) -> bool:
        plugin_id, kind, proc = fp
        if item.get("plugin_id") != plugin_id or item.get("kind") != kind:
            return False
        if proc is None:
            return True
        hint = _process_hint(item.get("where") if isinstance(item.get("where"), dict) else None) or _process_hint(
            item.get("contains") if isinstance(item.get("contains"), dict) else None
        )
        return isinstance(hint, str) and hint.strip().lower() == proc

    fingerprints = _known_fingerprints()
    discovery_items = [i for i in (discovery.get("items") or []) if isinstance(i, dict)]
    if fingerprints:
        filtered: list[dict[str, Any]] = []
        for item in discovery_items:
            if any(_matches_known(item, fp) for fp in fingerprints):
                continue
            filtered.append(item)
        discovery = dict(discovery)
        discovery["items"] = filtered
        # Keep status/summary consistent after landmark filtering.
        if filtered:
            discovery["status"] = "ok"
            discovery["summary"] = (
                f"Generated {len(filtered)} discovery recommendation(s) from plugin findings."
            )
        else:
            discovery["status"] = "none"
            discovery["summary"] = "No discovery recommendations generated from plugin findings."

    include_known = _include_known_recommendations()
    known = (
        _build_known_issue_recommendations(report, storage, run_dir=run_dir)
        if include_known
        else {
            "status": "suppressed",
            "summary": "Known-issue landmarks are excluded from recommendations by policy.",
            "items": [],
        }
    )
    combined = _dedupe_recommendations_text(
        [
            item
            for item in ((known.get("items") or []) if include_known else [])
            + (discovery.get("items") or [])
            if isinstance(item, dict)
        ]
    )
    summary = (
        "Combined known-issue pass gate and discovery recommendations."
        if include_known
        else "Discovery recommendations only (known-issue landmarks excluded by policy)."
    )
    explanations, coverage = _build_non_actionable_explanations(report, combined)
    return {
        "status": "ok",
        "summary": summary,
        "known": known,
        "discovery": discovery,
        "items": combined,
        "explanations": explanations,
        "actionability_coverage": coverage,
    }


def _build_executive_summary(report: dict[str, Any]) -> list[str]:
    plugins = report.get("plugins", {}) or {}
    queue_plugin = plugins.get("analysis_queue_delay_decomposition")
    if not isinstance(queue_plugin, dict):
        return []
    findings = queue_plugin.get("findings") or []
    qemail_stats = None
    impact = None
    scale = None
    for item in findings:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        if kind == "eligible_wait_process_stats" and item.get("process_norm") == "qemail":
            qemail_stats = item
        elif kind == "eligible_wait_impact":
            impact = item
        elif kind == "capacity_scale_model":
            scale = item

    lines: list[str] = []
    if qemail_stats and impact:
        q_gt = qemail_stats.get("eligible_wait_gt_hours_total")
        total_gt = impact.get("eligible_wait_gt_hours_total")
        runs_total = qemail_stats.get("runs_total")
        if isinstance(q_gt, (int, float)) and isinstance(total_gt, (int, float)) and total_gt:
            share = (float(q_gt) / float(total_gt)) * 100.0
            runs_text = f" across {int(runs_total):,} runs" if isinstance(runs_total, (int, float)) else ""
            lines.append(
                "QEMAIL is a major close-cycle drag: "
                f"{float(q_gt):.2f}h of >threshold eligible wait out of "
                f"{float(total_gt):.2f}h total ({share:.1f}%).{runs_text}"
            )

    if scale and _include_capacity_recommendations():
        base = scale.get("eligible_wait_gt_hours_without_target")
        modeled = scale.get("eligible_wait_gt_hours_modeled")
        if isinstance(base, (int, float)) and isinstance(modeled, (int, float)) and base:
            delta = float(base) - float(modeled)
            pct = (delta / float(base)) * 100.0 if base else 0.0
            lines.append(
                "QPEC+1 recommended (modeled): "
                f">threshold eligible wait drops from {float(base):.2f}h to "
                f"{float(modeled):.2f}h (Δ {delta:.2f}h, {pct:.1f}%)."
            )

    return lines


def _evaluate_known_issues(report: dict[str, Any]) -> list[dict[str, Any]]:
    known = report.get("known_issues")
    if not isinstance(known, dict):
        return []
    expected = known.get("expected_findings") or []
    if not isinstance(expected, list) or not expected:
        return []
    evaluations: list[dict[str, Any]] = []
    for issue in expected:
        if not isinstance(issue, dict):
            continue
        plugin_id = issue.get("plugin_id")
        kind = issue.get("kind")
        where = issue.get("where") if isinstance(issue.get("where"), dict) else None
        contains = issue.get("contains") if isinstance(issue.get("contains"), dict) else None
        min_count = issue.get("min_count")
        max_count = issue.get("max_count")
        title = issue.get("title") or issue.get("description") or ""
        label = title or f"{plugin_id or 'any'}:{kind or 'finding'}"

        findings = _collect_findings_for_plugin(report, plugin_id, kind)
        matched = [f for f in findings if _matches_expected(f, where, contains)]
        count = len(matched)
        process_hint = _process_hint(where)
        if count == 0:
            known_item = _known_recommendation_match(
                report, plugin_id, kind, process_hint
            )
            if isinstance(known_item, dict):
                matched = [known_item]
                observed = known_item.get("observed_count")
                if isinstance(observed, (int, float)):
                    count = int(observed)
                else:
                    count = 1

        status = "confirmed"
        if count == 0:
            status = "missing"
        if min_count is not None:
            try:
                if count < int(min_count):
                    status = "below_min"
            except (TypeError, ValueError):
                pass
        if max_count is not None:
            try:
                if count > int(max_count):
                    status = "over_limit"
            except (TypeError, ValueError):
                pass

        evaluations.append(
            {
                "issue": issue,
                "label": label,
                "plugin_id": plugin_id,
                "kind": kind,
                "where": where,
                "contains": contains,
                "min_count": min_count,
                "max_count": max_count,
                "matched": matched,
                "count": count,
                "status": status,
                "process_hint": process_hint,
            }
        )
    return evaluations


def _format_issue_value(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, int):
            return f"{value}"
        if float(value).is_integer():
            return f"{int(value)}"
        return f"{float(value):.{digits}f}"
    return str(value)


def _artifact_paths(report: dict[str, Any], plugin_id: str | None) -> list[str]:
    if not plugin_id:
        return []
    plugin = report.get("plugins", {}).get(plugin_id) or {}
    artifacts = plugin.get("artifacts") or []
    paths = []
    for artifact in artifacts:
        if isinstance(artifact, dict):
            path = artifact.get("path")
            if isinstance(path, str) and path:
                paths.append(path)
    return paths


def _metric_spec(kind: str | None) -> dict[str, Any]:
    return {
        "eligible_wait_process_stats": {
            "name": "Eligible wait > threshold (hours)",
            "definition": "Total hours of eligible wait above the threshold for the process.",
            "baseline_field": "eligible_wait_gt_hours_total",
            "observed_field": "eligible_wait_gt_hours_total",
            "denominator_field": "runs_total",
        },
        "eligible_wait_impact": {
            "name": "Eligible wait > threshold (hours)",
            "definition": "Total hours of eligible wait above the threshold (all runs).",
            "baseline_field": "eligible_wait_gt_hours_total",
            "observed_field": "eligible_wait_gt_hours_without_target",
            "denominator_field": "runs_total",
        },
        "capacity_scale_model": {
            "name": "Eligible wait > threshold (hours)",
            "definition": "Modeled eligible wait above the threshold after capacity scaling.",
            "baseline_field": "eligible_wait_gt_hours_without_target",
            "observed_field": "eligible_wait_gt_hours_modeled",
            "denominator_field": "runs_total",
        },
        "capacity_scaling": {
            "name": "Eligible wait (hours)",
            "definition": "Modeled eligible wait hours after capacity scaling.",
            "baseline_field": "baseline_wait_hours",
            "observed_field": "modeled_wait_hours",
            "denominator_field": "rows",
        },
        "close_cycle_capacity_model": {
            "name": "Close-cycle median duration (seconds)",
            "definition": "Modeled median close-cycle duration under added capacity.",
            "baseline_field": "baseline_median_sec",
            "observed_field": "modeled_median_sec",
            "denominator_field": "bucket_count",
        },
        "close_cycle_capacity_impact": {
            "name": "Close-cycle median duration (seconds)",
            "definition": "Measured close-cycle median duration shift.",
            "baseline_field": "baseline_median_sec",
            "observed_field": "modeled_median_sec",
            "denominator_field": "bucket_count",
        },
        "close_cycle_revenue_compression": {
            "name": "Close-cycle span (days)",
            "definition": "Median close-cycle span by revenue month.",
            "baseline_field": "baseline_span_days_median",
            "observed_field": "baseline_span_days_median",
            "denominator_field": "months",
        },
        "close_cycle_duration_shift": {
            "name": "Close-cycle median duration (seconds)",
            "definition": "Median close-cycle duration vs open-window duration for the process.",
            "baseline_field": "median_open",
            "observed_field": "median_close",
            "denominator_field": "close_count",
        },
        "upload_bkrvnu_linkage": {
            "name": "BKRVNU upload linkage coverage (%)",
            "definition": "Share of BKRVNU rows that can be matched to an XLSX upload by user + time window.",
            "baseline_field": "matched_user_pct",
            "observed_field": "matched_user_pct",
            "denominator_field": "bkrvnu_rows",
        },
        "process_counterfactual": {
            "name": "Over-threshold wait-to-start (hours)",
            "definition": "Modeled process wait above threshold after reducing to baseline quantile.",
            "baseline_field": "baseline_over_threshold_hours",
            "observed_field": "modeled_over_threshold_hours",
            "denominator_field": "runs_count",
        },
        "sequence_bottleneck": {
            "name": "Transition over-threshold wait (hours)",
            "definition": "Modeled transition gap above threshold after reducing to baseline quantile.",
            "baseline_field": "baseline_over_threshold_hours",
            "observed_field": "modeled_over_threshold_hours",
            "denominator_field": "transition_count",
        },
        "user_host_savings": {
            "name": "Over-threshold wait-to-start (hours)",
            "definition": "Modeled savings from rebalancing user/host to baseline wait level.",
            "baseline_field": "baseline_over_threshold_hours",
            "observed_field": "modeled_over_threshold_hours",
            "denominator_field": "runs_count",
        },
    }.get(kind or "", {"name": "Finding count", "definition": "Count of matching findings."})


def _denominator_text(item: dict[str, Any], report: dict[str, Any], spec: dict[str, Any]) -> str:
    field = spec.get("denominator_field")
    value = item.get(field) if isinstance(field, str) else None
    parts: list[str] = []
    if value is not None:
        parts.append(f"{field}={_format_issue_value(value)}")
    if "close_cycle_start_day" in item and "close_cycle_end_day" in item:
        parts.append(
            f"close_window=day{item.get('close_cycle_start_day')}-day{item.get('close_cycle_end_day')}"
        )
    if "close_window_mode" in item:
        parts.append(f"close_window_mode={item.get('close_window_mode')}")
    if not parts:
        rows = report.get("input", {}).get("rows")
        if rows is not None:
            parts.append(f"rows={rows}")
    return ", ".join(parts) if parts else "n/a"


def _impact_hours(item: dict[str, Any]) -> float:
    for field in (
        "delta_value",
        "delta_hours",
        "modeled_delta",
        "eligible_wait_gt_hours_total",
        "eligible_wait_hours_total",
        "baseline_wait_hours",
        "reduction_hours",
    ):
        value = item.get(field)
        if isinstance(value, (int, float)):
            return float(value)
    baseline_days = item.get("baseline_span_days_median")
    if isinstance(baseline_days, (int, float)):
        return float(baseline_days) * 24.0
    baseline_sec = item.get("baseline_median_sec")
    if isinstance(baseline_sec, (int, float)):
        return float(baseline_sec) / 3600.0
    return 0.0


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


def _item_evidence_row(item: dict[str, Any]) -> dict[str, Any]:
    evidence = item.get("evidence")
    if isinstance(evidence, list):
        for row in evidence:
            if isinstance(row, dict):
                return row
    if isinstance(evidence, dict):
        return evidence
    return {}


def _target_process_ids_for_item(item: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for bucket in (item, _item_evidence_row(item)):
        raw = bucket.get("target_process_ids") if isinstance(bucket, dict) else None
        if isinstance(raw, list):
            for value in raw:
                if isinstance(value, str) and value.strip():
                    out.append(value.strip().lower())
    process_norm = _item_process_norm(item)
    if process_norm:
        out.append(process_norm)
    deduped: list[str] = []
    seen: set[str] = set()
    for token in out:
        if not token or token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _scope_type_for_item(item: dict[str, Any]) -> str:
    targets = _target_process_ids_for_item(item)
    return "grouped_explicit" if len(targets) > 1 else "single_process"


def _primary_process_for_item(item: dict[str, Any]) -> str:
    targets = _target_process_ids_for_item(item)
    return targets[0] if targets else _item_process_norm(item)


def _safe_num(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _scope_class_for_item(item: dict[str, Any]) -> str:
    plugin_id = str(item.get("plugin_id") or "").strip().lower()
    kind = str(item.get("kind") or "").strip().lower()
    action_type = str(item.get("action_type") or item.get("action") or "").strip().lower()
    text = str(item.get("recommendation") or "").strip().lower()
    process_norm = _item_process_norm(item)

    if plugin_id.startswith("analysis_close_cycle_"):
        return "close_specific"
    if kind.startswith("close_cycle_") or "spillover" in kind:
        return "close_specific"
    if "close-cycle" in text or "month-end" in text or "eom" in text:
        return "close_specific"
    if process_norm in {"qemail", "qpec"} and action_type in {"tune_schedule", "add_server"}:
        return "close_specific"
    return "general"


def _parse_month_token(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if not value:
        return None
    for fmt in ("%Y-%m", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    match = re.search(r"(\d{4})-(\d{2})", value)
    if not match:
        return None
    try:
        return datetime.strptime(f"{match.group(1)}-{match.group(2)}", "%Y-%m")
    except ValueError:
        return None


def _item_month(item: dict[str, Any]) -> datetime | None:
    for key in ("close_month", "month", "revenue_month"):
        parsed = _parse_month_token(item.get(key))
        if parsed:
            return parsed
    evidence = item.get("evidence")
    if isinstance(evidence, list):
        for row in evidence:
            if not isinstance(row, dict):
                continue
            for key in ("close_month", "month", "revenue_month"):
                parsed = _parse_month_token(row.get(key))
                if parsed:
                    return parsed
    text = str(item.get("recommendation") or "")
    return _parse_month_token(text)


def _apply_recency_weight(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unknown_weight = _env_float(
        _RECENCY_UNKNOWN_WEIGHT_ENV,
        0.85,
        min_value=0.0,
        max_value=1.0,
    )
    decay_per_month = _env_float(
        _RECENCY_DECAY_PER_MONTH_ENV,
        0.25,
        min_value=0.0,
        max_value=10.0,
    )
    min_weight = _env_float(
        _RECENCY_MIN_WEIGHT_ENV,
        0.4,
        min_value=0.0,
        max_value=1.0,
    )
    months = [_item_month(item) for item in items]
    valid_months = [m for m in months if isinstance(m, datetime)]
    if not valid_months:
        for item in items:
            item["recency_weight"] = 1.0
        return items

    newest = max(valid_months)
    for item in items:
        month = _item_month(item)
        if not month:
            weight = unknown_weight
        else:
            month_delta = max(0, (newest.year - month.year) * 12 + (newest.month - month.month))
            weight = 1.0 / (1.0 + float(decay_per_month) * float(month_delta))
            if weight < min_weight:
                weight = min_weight
        item["recency_weight"] = round(float(weight), 4)
        relevance = item.get("relevance_score")
        if isinstance(relevance, (int, float)):
            item["relevance_score"] = float(relevance) * float(weight)
    return items


def _day_predicate_sql(col: str, *, start: int, end: int) -> tuple[str, list[Any]]:
    day_expr = f"CAST(SUBSTR({col}, 9, 2) AS INTEGER)"
    if start <= end:
        return f"({day_expr} BETWEEN ? AND ?)", [int(start), int(end)]
    return f"(({day_expr} >= ?) OR ({day_expr} <= ?))", [int(start), int(end)]


def _window_predicate_sql(
    col: str,
    ranges: list[tuple[datetime, datetime]],
) -> tuple[str, list[Any]]:
    if not ranges:
        return "0", []
    clauses: list[str] = []
    params: list[Any] = []
    for start, end in ranges:
        clauses.append(f"({col} >= ? AND {col} <= ?)")
        params.append(start.strftime("%Y-%m-%d %H:%M:%S"))
        params.append(end.strftime("%Y-%m-%d %H:%M:%S"))
    return "(" + " OR ".join(clauses) + ")", params


def _resolved_accounting_windows(
    report: dict[str, Any],
    storage: Storage | None,
    *,
    run_dir: Path | None,
    table_name: str,
    timestamp_col: str,
) -> list[AccountingWindow]:
    if run_dir is not None:
        from_run = load_accounting_windows_from_run(run_dir)
        if from_run:
            return from_run
    if storage is None:
        return []
    try:
        with storage.connection() as conn:
            rows = conn.execute(
                f"SELECT {timestamp_col} AS ts FROM {table_name} WHERE {timestamp_col} IS NOT NULL LIMIT 250000"
            ).fetchall()
    except Exception:
        return []
    if not rows:
        return []
    series = [row["ts"] for row in rows if row["ts"] is not None]
    return infer_accounting_windows_from_timestamps(series)


def _duration_baselines(
    report: dict[str, Any], storage: Storage | None, run_dir: Path | None = None
) -> dict[str, float] | None:
    if storage is None:
        return None
    _, table_name, mapping = _dataset_context(report, storage)
    if not table_name or not mapping:
        return None
    proc_col = mapping.get("PROCESS_ID")
    start_col = mapping.get("START_DT")
    end_col = mapping.get("END_DT")
    queue_col = mapping.get("QUEUE_DT") or start_col
    if not proc_col or not start_col or not end_col or not queue_col:
        return None
    windows = _resolved_accounting_windows(
        report, storage, run_dir=run_dir, table_name=table_name, timestamp_col=queue_col
    )
    accounting_pred: str
    accounting_params: list[Any]
    close_static_pred: str
    close_static_params: list[Any]
    close_dynamic_pred: str
    close_dynamic_params: list[Any]
    if windows:
        accounting_pred, accounting_params = _window_predicate_sql(
            queue_col, window_ranges(windows, kind="accounting_month")
        )
        close_static_pred, close_static_params = _window_predicate_sql(
            queue_col, window_ranges(windows, kind="close_static")
        )
        close_dynamic_pred, close_dynamic_params = _window_predicate_sql(
            queue_col, window_ranges(windows, kind="close_dynamic")
        )
    else:
        accounting_pred, accounting_params = "1", []
        close_start, close_end = _close_cycle_bounds(report)
        if not isinstance(close_start, int) or not isinstance(close_end, int):
            close_start, close_end = 1, 31
        close_static_pred, close_static_params = _day_predicate_sql(queue_col, start=close_start, end=close_end)
        close_dynamic_pred, close_dynamic_params = close_static_pred, list(close_static_params)
    duration_expr = (
        f"(CASE WHEN {start_col} IS NOT NULL AND {end_col} IS NOT NULL "
        f"AND julianday({end_col}) >= julianday({start_col}) "
        f"THEN (julianday({end_col}) - julianday({start_col})) * 24.0 ELSE 0.0 END)"
    )
    query = f"""
    SELECT
        COALESCE(SUM({duration_expr}), 0.0) AS total_duration_hours,
        COALESCE(SUM(CASE WHEN {accounting_pred} THEN {duration_expr} ELSE 0.0 END), 0.0) AS accounting_duration_hours,
        COALESCE(SUM(CASE WHEN {close_static_pred} THEN {duration_expr} ELSE 0.0 END), 0.0) AS close_static_duration_hours,
        COALESCE(SUM(CASE WHEN {close_dynamic_pred} THEN {duration_expr} ELSE 0.0 END), 0.0) AS close_dynamic_duration_hours
    FROM {table_name}
    """
    params = [*accounting_params, *close_static_params, *close_dynamic_params]
    with storage.connection() as conn:
        row = conn.execute(query, params).fetchone()
    if not row:
        return None
    total_hours = float(row["total_duration_hours"] or 0.0)
    accounting_hours = float(row["accounting_duration_hours"] or 0.0)
    close_static_hours = float(row["close_static_duration_hours"] or 0.0)
    close_dynamic_hours = float(row["close_dynamic_duration_hours"] or 0.0)
    if accounting_hours <= 0.0:
        accounting_hours = total_hours
    return {
        "general_basis_hours": accounting_hours,
        "close_basis_hours": close_dynamic_hours,
        "accounting_month_basis_hours": accounting_hours,
        "close_static_basis_hours": close_static_hours,
        "close_dynamic_basis_hours": close_dynamic_hours,
        "observation_basis_hours": total_hours,
    }


def _process_removal_model(
    report: dict[str, Any], storage: Storage | None, process_norm: str, run_dir: Path | None = None
) -> dict[str, float] | None:
    if storage is None:
        return None
    proc = str(process_norm or "").strip().lower()
    if not proc:
        return None
    _, table_name, mapping = _dataset_context(report, storage)
    if not table_name or not mapping:
        return None
    proc_col = mapping.get("PROCESS_ID")
    start_col = mapping.get("START_DT")
    end_col = mapping.get("END_DT")
    queue_col = mapping.get("QUEUE_DT") or start_col
    if not proc_col or not start_col or not end_col or not queue_col:
        return None
    windows = _resolved_accounting_windows(
        report, storage, run_dir=run_dir, table_name=table_name, timestamp_col=queue_col
    )
    if windows:
        accounting_pred, accounting_params = _window_predicate_sql(
            queue_col, window_ranges(windows, kind="accounting_month")
        )
        close_static_pred, close_static_params = _window_predicate_sql(
            queue_col, window_ranges(windows, kind="close_static")
        )
        close_dynamic_pred, close_dynamic_params = _window_predicate_sql(
            queue_col, window_ranges(windows, kind="close_dynamic")
        )
    else:
        accounting_pred, accounting_params = "1", []
        close_start, close_end = _close_cycle_bounds(report)
        if not isinstance(close_start, int) or not isinstance(close_end, int):
            close_start, close_end = 1, 31
        close_static_pred, close_static_params = _day_predicate_sql(queue_col, start=close_start, end=close_end)
        close_dynamic_pred, close_dynamic_params = close_static_pred, list(close_static_params)
    duration_expr = (
        f"(CASE WHEN {start_col} IS NOT NULL AND {end_col} IS NOT NULL "
        f"AND julianday({end_col}) >= julianday({start_col}) "
        f"THEN (julianday({end_col}) - julianday({start_col})) * 24.0 ELSE 0.0 END)"
    )
    query = f"""
    SELECT
        COALESCE(SUM({duration_expr}), 0.0) AS total_duration_hours,
        COALESCE(SUM(CASE WHEN LOWER({proc_col}) = ? THEN {duration_expr} ELSE 0.0 END), 0.0) AS process_duration_hours,
        COALESCE(SUM(CASE WHEN {accounting_pred} THEN {duration_expr} ELSE 0.0 END), 0.0) AS accounting_duration_hours,
        COALESCE(SUM(CASE WHEN LOWER({proc_col}) = ? AND {accounting_pred} THEN {duration_expr} ELSE 0.0 END), 0.0) AS process_accounting_duration_hours,
        COALESCE(SUM(CASE WHEN {close_static_pred} THEN {duration_expr} ELSE 0.0 END), 0.0) AS close_static_duration_hours,
        COALESCE(SUM(CASE WHEN LOWER({proc_col}) = ? AND {close_static_pred} THEN {duration_expr} ELSE 0.0 END), 0.0) AS process_close_static_duration_hours,
        COALESCE(SUM(CASE WHEN {close_dynamic_pred} THEN {duration_expr} ELSE 0.0 END), 0.0) AS close_dynamic_duration_hours,
        COALESCE(SUM(CASE WHEN LOWER({proc_col}) = ? AND {close_dynamic_pred} THEN {duration_expr} ELSE 0.0 END), 0.0) AS process_close_dynamic_duration_hours
    FROM {table_name}
    """
    params = [
        proc,
        *accounting_params,
        proc,
        *accounting_params,
        *close_static_params,
        proc,
        *close_static_params,
        *close_dynamic_params,
        proc,
        *close_dynamic_params,
    ]
    with storage.connection() as conn:
        row = conn.execute(query, params).fetchone()
    if not row:
        return None
    total_hours = float(row["total_duration_hours"] or 0.0)
    process_hours = float(row["process_duration_hours"] or 0.0)
    accounting_hours = float(row["accounting_duration_hours"] or 0.0)
    process_accounting_hours = float(row["process_accounting_duration_hours"] or 0.0)
    close_static_hours = float(row["close_static_duration_hours"] or 0.0)
    process_close_static_hours = float(row["process_close_static_duration_hours"] or 0.0)
    close_dynamic_hours = float(row["close_dynamic_duration_hours"] or 0.0)
    process_close_dynamic_hours = float(row["process_close_dynamic_duration_hours"] or 0.0)
    if accounting_hours <= 0.0:
        accounting_hours = total_hours
        process_accounting_hours = process_hours
    if accounting_hours <= 0.0 and close_dynamic_hours <= 0.0:
        return None
    accounting_pct = (
        (process_accounting_hours / accounting_hours) * 100.0 if accounting_hours > 0.0 else None
    )
    close_static_pct = (
        (process_close_static_hours / close_static_hours) * 100.0 if close_static_hours > 0.0 else None
    )
    close_dynamic_pct = (
        (process_close_dynamic_hours / close_dynamic_hours) * 100.0 if close_dynamic_hours > 0.0 else None
    )
    return {
        "general_basis_hours": accounting_hours,
        "general_delta_hours": process_accounting_hours,
        "general_modeled_percent": float(accounting_pct) if isinstance(accounting_pct, float) else None,
        "close_basis_hours": close_dynamic_hours,
        "close_delta_hours": process_close_dynamic_hours,
        "close_modeled_percent": float(close_dynamic_pct) if isinstance(close_dynamic_pct, float) else None,
        "accounting_month_basis_hours": accounting_hours,
        "accounting_month_delta_hours": process_accounting_hours,
        "accounting_month_modeled_percent": float(accounting_pct) if isinstance(accounting_pct, float) else None,
        "close_static_basis_hours": close_static_hours,
        "close_static_delta_hours": process_close_static_hours,
        "close_static_modeled_percent": float(close_static_pct) if isinstance(close_static_pct, float) else None,
        "close_dynamic_basis_hours": close_dynamic_hours,
        "close_dynamic_delta_hours": process_close_dynamic_hours,
        "close_dynamic_modeled_percent": float(close_dynamic_pct) if isinstance(close_dynamic_pct, float) else None,
        "observation_basis_hours": total_hours,
    }


def _kona_modeled_percent_for_process(report: dict[str, Any], process_norm: str) -> float | None:
    proc = str(process_norm or "").strip().lower()
    if not proc:
        return None
    plugins = report.get("plugins")
    if not isinstance(plugins, dict):
        return None
    verifier = plugins.get("analysis_ebm_action_verifier_v1")
    if not isinstance(verifier, dict):
        return None
    findings = verifier.get("findings")
    if not isinstance(findings, list):
        return None
    best: float | None = None
    for item in findings:
        if not isinstance(item, dict):
            continue
        lever_id = str(item.get("lever_id") or "").strip().lower()
        target = str(item.get("target") or "").strip().lower()
        what = str(item.get("what") or "").strip().lower()
        if proc == "qemail":
            if (lever_id != "tune_schedule_qemail_frequency_v1") and ("qemail" not in target) and ("qemail" not in what):
                continue
        pct: float | None = None
        delta = item.get("delta_energy")
        before = item.get("energy_before")
        if isinstance(delta, (int, float)) and isinstance(before, (int, float)) and float(before) > 0.0:
            pct = (float(delta) / float(before)) * 100.0
        if not isinstance(pct, (int, float)) and isinstance(item.get("delta_value"), (int, float)):
            unit = str(item.get("unit") or "").strip().lower()
            raw = float(item.get("delta_value") or 0.0)
            if unit in {"percent", "pct", "%"}:
                pct = raw * 100.0 if 0.0 <= raw <= 1.0 else raw
        if isinstance(pct, (int, float)) and math.isfinite(float(pct)):
            val = max(0.0, min(100.0, float(pct)))
            if (best is None) or (val > best):
                best = val
    return best


def _queue_wait_hours_for_process(report: dict[str, Any], process_norm: str) -> float | None:
    proc = str(process_norm or "").strip().lower()
    if not proc:
        return None
    plugins = report.get("plugins")
    if not isinstance(plugins, dict):
        return None
    payload = plugins.get("analysis_queue_delay_decomposition")
    if not isinstance(payload, dict):
        return None
    findings = payload.get("findings")
    if not isinstance(findings, list):
        return None
    for item in findings:
        if not isinstance(item, dict):
            continue
        if item.get("kind") != "eligible_wait_process_stats":
            continue
        proc_item = str(item.get("process_norm") or item.get("process") or "").strip().lower()
        if proc_item != proc:
            continue
        value = item.get("eligible_wait_hours_total")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _queue_wait_hours_total(report: dict[str, Any]) -> float | None:
    plugins = report.get("plugins")
    if not isinstance(plugins, dict):
        return None
    payload = plugins.get("analysis_queue_delay_decomposition")
    if not isinstance(payload, dict):
        return None
    findings = payload.get("findings")
    if not isinstance(findings, list):
        return None
    for item in findings:
        if not isinstance(item, dict):
            continue
        if item.get("kind") != "eligible_wait_impact":
            continue
        for key in ("eligible_wait_hours_total", "eligible_wait_gt_hours_total"):
            value = item.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value)
    return None


def _item_metrics_payload(item: dict[str, Any]) -> dict[str, Any]:
    evidence = item.get("evidence")
    if isinstance(evidence, list):
        for row in evidence:
            if not isinstance(row, dict):
                continue
            metrics = row.get("metrics")
            if isinstance(metrics, dict):
                return metrics
    if isinstance(evidence, dict):
        metrics = evidence.get("metrics")
        if isinstance(metrics, dict):
            return metrics
    return {}


def _modeled_rollups(
    *,
    delta_hours: float | None,
    scope_class: str,
) -> dict[str, float | None]:
    if not isinstance(delta_hours, (int, float)) or float(delta_hours) <= 0.0:
        return {
            "close_cycle": None,
            "month": None,
            "annualized": None,
        }
    close_cycle = float(delta_hours) if scope_class == "close_specific" else None
    month = float(delta_hours)
    annualized = float(month) * float(_close_cycles_per_year())
    return {
        "close_cycle": close_cycle,
        "month": month,
        "annualized": annualized,
    }


def _attach_client_value_metrics(
    enriched: dict[str, Any],
    *,
    scope_class: str,
) -> None:
    evidence = _item_evidence_row(enriched)
    targets = _target_process_ids_for_item(enriched)
    scope_type = _scope_type_for_item(enriched)
    primary_process = _primary_process_for_item(enriched)
    action_type = str(enriched.get("action_type") or enriched.get("action") or "").strip().lower()
    enriched["scope_type"] = scope_type
    enriched["primary_process_id"] = primary_process or None
    enriched["target_process_ids"] = targets
    enriched["target_process_count"] = int(len(targets))

    delta_hours = _safe_num(enriched.get("modeled_delta_hours"))
    rollups = _modeled_rollups(delta_hours=delta_hours, scope_class=scope_class)
    enriched["modeled_close_hours_saved_cycle"] = rollups["close_cycle"]
    enriched["modeled_close_hours_saved_month"] = rollups["month"]
    enriched["modeled_close_hours_saved_annualized"] = rollups["annualized"]

    estimated_calls_reduced = _safe_num(
        enriched.get("estimated_calls_reduced")
        if enriched.get("estimated_calls_reduced") is not None
        else evidence.get("estimated_calls_reduced")
    )
    if not isinstance(estimated_calls_reduced, (int, float)):
        runs_with_key = _safe_num(enriched.get("runs_with_key"))
        if not isinstance(runs_with_key, (int, float)):
            runs_with_key = _safe_num(evidence.get("runs_with_key"))
        if isinstance(runs_with_key, (int, float)) and float(runs_with_key) > 1.0:
            estimated_calls_reduced = float(runs_with_key) - 1.0

    top_user_share = _safe_num(
        enriched.get("top_user_run_share")
        if enriched.get("top_user_run_share") is not None
        else evidence.get("top_user_run_share")
    )
    if not isinstance(top_user_share, (int, float)):
        top_user_runs = _safe_num(
            enriched.get("top_user_runs")
            if enriched.get("top_user_runs") is not None
            else evidence.get("top_user_runs")
        )
        user_runs_total = _safe_num(
            enriched.get("user_runs_total")
            if enriched.get("user_runs_total") is not None
            else evidence.get("user_runs_total")
        )
        if (
            isinstance(top_user_runs, (int, float))
            and isinstance(user_runs_total, (int, float))
            and float(user_runs_total) > 0.0
        ):
            top_user_share = max(
                0.0,
                min(1.0, float(top_user_runs) / float(user_runs_total)),
            )

    user_touches_reduced = None
    if isinstance(estimated_calls_reduced, (int, float)) and float(estimated_calls_reduced) > 0.0:
        if isinstance(top_user_share, (int, float)) and float(top_user_share) > 0.0:
            user_touches_reduced = float(estimated_calls_reduced) * float(top_user_share)
        else:
            user_touches_reduced = float(estimated_calls_reduced)
    enriched["modeled_user_touches_reduced"] = user_touches_reduced
    enriched["modeled_user_runs_reduced"] = user_touches_reduced
    manual_run_reduction_pct = None
    runs_with_key = _safe_num(
        enriched.get("runs_with_key")
        if enriched.get("runs_with_key") is not None
        else evidence.get("runs_with_key")
    )
    if (
        isinstance(user_touches_reduced, (int, float))
        and isinstance(runs_with_key, (int, float))
        and float(runs_with_key) > 0.0
    ):
        manual_run_reduction_pct = max(
            0.0,
            min(100.0, (float(user_touches_reduced) / float(runs_with_key)) * 100.0),
        )
    enriched["modeled_manual_run_reduction_pct"] = (
        float(manual_run_reduction_pct)
        if isinstance(manual_run_reduction_pct, (int, float))
        else None
    )

    user_hours_close = None
    if isinstance(delta_hours, (int, float)) and float(delta_hours) > 0.0:
        if isinstance(top_user_share, (int, float)) and float(top_user_share) > 0.0:
            user_hours_close = float(delta_hours) * float(top_user_share)
        elif isinstance(user_touches_reduced, (int, float)) and float(user_touches_reduced) > 0.0:
            user_hours_close = float(delta_hours)
    user_rollups = _modeled_rollups(delta_hours=user_hours_close, scope_class=scope_class)
    enriched["modeled_user_hours_saved_close_cycle"] = user_rollups["close_cycle"]
    enriched["modeled_user_hours_saved_month"] = user_rollups["month"]
    enriched["modeled_user_hours_saved_annualized"] = user_rollups["annualized"]

    top_user_redacted = str(
        enriched.get("top_user_redacted")
        if enriched.get("top_user_redacted") is not None
        else evidence.get("top_user_redacted")
        or ""
    ).strip()
    if top_user_redacted:
        enriched["affected_user_primary"] = top_user_redacted
    distinct_users = _safe_num(
        enriched.get("distinct_users")
        if enriched.get("distinct_users") is not None
        else evidence.get("distinct_users")
    )
    if isinstance(distinct_users, (int, float)):
        enriched["affected_user_count"] = int(max(0, int(float(distinct_users))))

    metrics = _item_metrics_payload(enriched)
    contention_pct = None
    p95_before = _safe_num(metrics.get("eligible_wait_p95_s"))
    p95_modeled = _safe_num(metrics.get("modeled_wait_p95_s"))
    if (
        isinstance(p95_before, (int, float))
        and isinstance(p95_modeled, (int, float))
        and float(p95_before) > 0.0
        and float(p95_modeled) >= 0.0
        and float(p95_modeled) < float(p95_before)
    ):
        contention_pct = ((float(p95_before) - float(p95_modeled)) / float(p95_before)) * 100.0
    if not isinstance(contention_pct, (int, float)) and scope_class == "close_specific":
        modeled_pct = _safe_num(enriched.get("modeled_percent"))
        if isinstance(modeled_pct, (int, float)) and float(modeled_pct) > 0.0:
            contention_pct = float(modeled_pct)
    enriched["modeled_contention_reduction_pct_close"] = (
        float(contention_pct)
        if isinstance(contention_pct, (int, float)) and float(contention_pct) > 0.0
        else None
    )
    enriched["modeled_queue_wait_hours_saved_close"] = (
        float(rollups["close_cycle"])
        if isinstance(rollups["close_cycle"], (int, float)) and float(rollups["close_cycle"]) > 0.0
        else None
    )

    confidence = _safe_num(enriched.get("confidence"))
    if not isinstance(confidence, (int, float)):
        confidence = _safe_num(enriched.get("confidence_weight"))
    if not isinstance(confidence, (int, float)):
        confidence = 0.5
    confidence = max(0.0, min(1.0, float(confidence)))
    enriched["metric_confidence"] = confidence

    user_effort_score = (
        max(0.0, float(user_rollups["close_cycle"] or 0.0))
        + math.log1p(max(0.0, float(user_touches_reduced or 0.0)))
    )
    close_window_score = (
        max(0.0, float(rollups["close_cycle"] or 0.0))
        + (0.35 * max(0.0, float(rollups["month"] or 0.0)))
    )
    server_contention_score = (
        max(0.0, float(enriched.get("modeled_queue_wait_hours_saved_close") or 0.0))
        + (max(0.0, float(enriched.get("modeled_contention_reduction_pct_close") or 0.0)) / 100.0)
    )
    targeting_bonus = 1.0 if scope_type == "single_process" else 0.35
    ambiguity_penalty = 0.0
    if scope_type != "single_process":
        ambiguity_penalty = 0.20 + (0.10 * max(0.0, float(len(targets) - 1)))

    value_score = (
        (0.45 * user_effort_score)
        + (0.30 * close_window_score)
        + (0.15 * server_contention_score)
        + (0.10 * confidence)
        + (0.20 * targeting_bonus)
        - ambiguity_penalty
    )
    enriched["value_components"] = {
        "user_effort_score": float(user_effort_score),
        "close_window_score": float(close_window_score),
        "server_contention_score": float(server_contention_score),
        "confidence_score": float(confidence),
        "targeting_bonus": float(targeting_bonus),
        "ambiguity_penalty": float(ambiguity_penalty),
    }
    enriched["client_value_score"] = float(value_score)
    enriched["value_score_v2"] = float(value_score)
    modeled_efficiency_gain_pct = _safe_num(enriched.get("efficiency_gain_pct_close_dynamic"))
    if not isinstance(modeled_efficiency_gain_pct, (int, float)):
        modeled_efficiency_gain_pct = _safe_num(enriched.get("modeled_percent"))
    if not isinstance(modeled_efficiency_gain_pct, (int, float)):
        modeled_efficiency_gain_pct = _safe_num(enriched.get("modeled_contention_reduction_pct_close"))
    enriched["modeled_efficiency_gain_pct"] = (
        float(modeled_efficiency_gain_pct)
        if isinstance(modeled_efficiency_gain_pct, (int, float))
        and float(modeled_efficiency_gain_pct) > 0.0
        else None
    )
    if action_type in {"add_server", "tune_schedule"} or primary_process in {"qpec", "qemail"}:
        enriched["opportunity_class"] = "server_capacity"
    elif action_type in {
        "batch_input",
        "batch_input_refactor",
        "batch_group_candidate",
        "batch_or_cache",
        "dedupe_or_cache",
        "throttle_or_dedupe",
    }:
        enriched["opportunity_class"] = "manual_automation"
    else:
        enriched["opportunity_class"] = "process_efficiency"


def _enrich_recommendation_item(
    item: dict[str, Any],
    report: dict[str, Any],
    baselines: dict[str, float] | None,
    qemail_model: dict[str, float] | None,
) -> dict[str, Any]:
    enriched = dict(item)
    scope_class = _scope_class_for_item(item)
    enriched["scope_class"] = scope_class
    process_norm = _item_process_norm(item)
    action_type = str(enriched.get("action_type") or enriched.get("action") or "").strip().lower()

    basis_hours: float | None = None
    delta_hours: float | None = None
    modeled_percent: float | None = None
    not_modeled_reason: str | None = None
    unit = str(enriched.get("unit") or "").strip().lower()

    if process_norm == "qemail" and isinstance(qemail_model, dict):
        enriched["modeled_general_percent"] = qemail_model.get("general_modeled_percent")
        enriched["modeled_close_percent"] = qemail_model.get("close_modeled_percent")
        if scope_class == "close_specific":
            basis_hours = qemail_model.get("close_basis_hours")
            delta_hours = qemail_model.get("close_delta_hours")
            modeled_percent = qemail_model.get("close_modeled_percent")
        else:
            basis_hours = qemail_model.get("general_basis_hours")
            delta_hours = qemail_model.get("general_delta_hours")
            modeled_percent = qemail_model.get("general_modeled_percent")
    if process_norm == "qemail":
        kona_pct = _kona_modeled_percent_for_process(report, "qemail")
        if isinstance(kona_pct, (int, float)) and math.isfinite(float(kona_pct)):
            kona_val = max(0.0, min(100.0, float(kona_pct)))
            prev_general = enriched.get("modeled_general_percent")
            prev_close = enriched.get("modeled_close_percent")
            if not isinstance(prev_general, (int, float)) or float(prev_general) < kona_val:
                enriched["modeled_general_percent"] = kona_val
            if not isinstance(prev_close, (int, float)) or float(prev_close) < kona_val:
                enriched["modeled_close_percent"] = kona_val
            if not isinstance(modeled_percent, (int, float)) or float(modeled_percent) < kona_val:
                modeled_percent = kona_val

    if not isinstance(basis_hours, (int, float)):
        proc_wait_basis = _queue_wait_hours_for_process(report, process_norm)
        if isinstance(proc_wait_basis, (int, float)) and float(proc_wait_basis) > 0.0:
            basis_hours = float(proc_wait_basis)
    if (
        not isinstance(basis_hours, (int, float))
        and action_type == "add_server"
    ):
        total_wait_basis = _queue_wait_hours_total(report)
        if isinstance(total_wait_basis, (int, float)) and float(total_wait_basis) > 0.0:
            basis_hours = float(total_wait_basis)

    if not isinstance(modeled_percent, (int, float)):
        pct_hint = enriched.get("modeled_percent_hint")
        if isinstance(pct_hint, (int, float)):
            modeled_percent = float(pct_hint)
        elif unit in {"percent", "pct", "%"} and isinstance(enriched.get("modeled_delta"), (int, float)):
            raw_pct = float(enriched.get("modeled_delta") or 0.0)
            modeled_percent = raw_pct * 100.0 if 0.0 <= raw_pct <= 1.0 else raw_pct
        elif unit in {"percent", "pct", "%"} and isinstance(enriched.get("delta_value"), (int, float)):
            raw_pct = float(enriched.get("delta_value") or 0.0)
            modeled_percent = raw_pct * 100.0 if 0.0 <= raw_pct <= 1.0 else raw_pct
        else:
            metrics = _item_metrics_payload(enriched)
            p95_before = metrics.get("eligible_wait_p95_s")
            p95_modeled = metrics.get("modeled_wait_p95_s")
            if (
                isinstance(p95_before, (int, float))
                and isinstance(p95_modeled, (int, float))
                and float(p95_before) > 0.0
                and float(p95_modeled) >= 0.0
                and float(p95_modeled) < float(p95_before)
            ):
                modeled_percent = (
                    (float(p95_before) - float(p95_modeled)) / float(p95_before)
                ) * 100.0

    if not isinstance(delta_hours, (int, float)):
        if isinstance(enriched.get("modeled_delta"), (int, float)):
            delta_hours = float(enriched.get("modeled_delta") or 0.0)
        elif isinstance(enriched.get("impact_hours"), (int, float)):
            impact_hours_val = float(enriched.get("impact_hours") or 0.0)
            if not (unit in {"percent", "pct", "%"} and impact_hours_val == 0.0):
                delta_hours = impact_hours_val
        elif isinstance(enriched.get("expected_delta_seconds"), (int, float)):
            delta_hours = float(enriched.get("expected_delta_seconds") or 0.0) / 3600.0
        elif isinstance(enriched.get("expected_delta_ms"), (int, float)):
            delta_hours = float(enriched.get("expected_delta_ms") or 0.0) / 3_600_000.0

    if not isinstance(basis_hours, (int, float)):
        if isinstance(baselines, dict):
            key = "close_basis_hours" if scope_class == "close_specific" else "general_basis_hours"
            basis_val = baselines.get(key)
            if isinstance(basis_val, (int, float)):
                basis_hours = float(basis_val)

    if isinstance(modeled_percent, (int, float)) and not isinstance(delta_hours, (int, float)):
        if isinstance(basis_hours, (int, float)) and float(basis_hours) > 0.0:
            delta_hours = (float(modeled_percent) / 100.0) * float(basis_hours)

    if not isinstance(modeled_percent, (int, float)):
        if isinstance(delta_hours, (int, float)) and isinstance(basis_hours, (int, float)):
            if float(basis_hours) > 0.0 and float(delta_hours) >= 0.0:
                modeled_percent = max(0.0, min(100.0, (float(delta_hours) / float(basis_hours)) * 100.0))
            else:
                not_modeled_reason = "basis_hours_not_positive"
        else:
            not_modeled_reason = "insufficient_modeled_inputs"

    if isinstance(delta_hours, (int, float)) and float(delta_hours) <= 0.0:
        delta_hours = None
    if isinstance(modeled_percent, (int, float)) and float(modeled_percent) <= 0.0:
        modeled_percent = None

    enriched["modeled_basis_hours"] = float(basis_hours) if isinstance(basis_hours, (int, float)) else None
    enriched["modeled_delta_hours"] = float(delta_hours) if isinstance(delta_hours, (int, float)) else None
    if isinstance(enriched.get("modeled_delta_hours"), (int, float)):
        enriched["modeled_delta"] = float(enriched["modeled_delta_hours"])
        enriched["impact_hours"] = float(enriched["modeled_delta_hours"])
    enriched["modeled_percent"] = float(modeled_percent) if isinstance(modeled_percent, (int, float)) else None
    if isinstance(enriched.get("modeled_delta_hours"), (int, float)):
        enriched["not_modeled_reason"] = None
    elif enriched["modeled_percent"] is None:
        enriched["not_modeled_reason"] = not_modeled_reason or "insufficient_modeled_inputs"
    else:
        enriched["not_modeled_reason"] = None

    # Required comparable window metrics for every recommendation row.
    acct_basis = None
    close_static_basis = None
    close_dynamic_basis = None
    acct_delta = None
    close_static_delta = None
    close_dynamic_delta = None

    if isinstance(process_norm, str) and process_norm == "qemail" and isinstance(qemail_model, dict):
        acct_basis = _safe_num(qemail_model.get("accounting_month_basis_hours"))
        close_static_basis = _safe_num(qemail_model.get("close_static_basis_hours"))
        close_dynamic_basis = _safe_num(qemail_model.get("close_dynamic_basis_hours"))
        acct_delta = _safe_num(qemail_model.get("accounting_month_delta_hours"))
        close_static_delta = _safe_num(qemail_model.get("close_static_delta_hours"))
        close_dynamic_delta = _safe_num(qemail_model.get("close_dynamic_delta_hours"))

    if not isinstance(acct_basis, (int, float)) and isinstance(baselines, dict):
        acct_basis = _safe_num(baselines.get("accounting_month_basis_hours"))
    if not isinstance(close_static_basis, (int, float)) and isinstance(baselines, dict):
        close_static_basis = _safe_num(baselines.get("close_static_basis_hours"))
    if not isinstance(close_dynamic_basis, (int, float)) and isinstance(baselines, dict):
        close_dynamic_basis = _safe_num(baselines.get("close_dynamic_basis_hours"))

    base_delta = _safe_num(delta_hours)
    if not isinstance(acct_delta, (int, float)):
        if isinstance(base_delta, (int, float)):
            acct_delta = float(base_delta)
    if not isinstance(close_dynamic_delta, (int, float)):
        if isinstance(base_delta, (int, float)):
            if scope_class == "close_specific":
                close_dynamic_delta = float(base_delta)
            elif isinstance(acct_basis, (int, float)) and isinstance(close_dynamic_basis, (int, float)) and float(acct_basis) > 0.0:
                close_dynamic_delta = float(base_delta) * max(0.0, float(close_dynamic_basis) / float(acct_basis))
    if not isinstance(close_static_delta, (int, float)):
        if isinstance(base_delta, (int, float)):
            if scope_class == "close_specific":
                close_static_delta = float(base_delta)
            elif isinstance(acct_basis, (int, float)) and isinstance(close_static_basis, (int, float)) and float(acct_basis) > 0.0:
                close_static_delta = float(base_delta) * max(0.0, float(close_static_basis) / float(acct_basis))

    for key, basis_value, delta_value in (
        ("accounting_month", acct_basis, acct_delta),
        ("close_static", close_static_basis, close_static_delta),
        ("close_dynamic", close_dynamic_basis, close_dynamic_delta),
    ):
        delta_field = f"delta_hours_{key}"
        eff_field = f"efficiency_gain_{key}"
        eff_pct_field = f"efficiency_gain_pct_{key}"
        na_field = f"na_reason_{key}"
        if isinstance(delta_value, (int, float)) and float(delta_value) > 0.0:
            enriched[delta_field] = float(delta_value)
        else:
            enriched[delta_field] = None

        if (
            isinstance(basis_value, (int, float))
            and float(basis_value) > 0.0
            and isinstance(delta_value, (int, float))
            and float(delta_value) >= 0.0
        ):
            eff = max(0.0, float(delta_value) / float(basis_value))
            enriched[eff_field] = float(eff)
            enriched[eff_pct_field] = float(eff * 100.0)
            enriched[na_field] = None
        else:
            enriched[eff_field] = None
            enriched[eff_pct_field] = None
            if not isinstance(basis_value, (int, float)) or float(basis_value) <= 0.0:
                enriched[na_field] = "missing_window_basis"
            elif not isinstance(delta_value, (int, float)):
                enriched[na_field] = enriched.get("not_modeled_reason") or "insufficient_modeled_inputs"
            else:
                enriched[na_field] = "insufficient_modeled_inputs"

    if isinstance(enriched.get("efficiency_gain_pct_close_dynamic"), (int, float)):
        enriched["modeled_efficiency_gain_pct"] = float(enriched["efficiency_gain_pct_close_dynamic"])

    _attach_client_value_metrics(enriched, scope_class=scope_class)
    return enriched


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


def _issue_cards(report: dict[str, Any]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for evaluation in _evaluate_known_issues(report):
        matched = evaluation.get("matched") or []
        item = matched[0] if matched else {}
        spec = _metric_spec(evaluation.get("kind"))
        metric_name = spec.get("name", "Metric")
        definition = spec.get("definition", "")
        baseline_field = spec.get("baseline_field")
        observed_field = spec.get("observed_field")
        baseline_val = item.get(baseline_field) if baseline_field else None
        observed_val = item.get(observed_field) if observed_field else None
        target_val = None
        if evaluation.get("kind") == "capacity_scale_model" and isinstance(
            baseline_val, (int, float)
        ):
            scale = item.get("scale_factor")
            if isinstance(scale, (int, float)):
                target_val = float(baseline_val) * float(scale)
        if evaluation.get("kind") == "capacity_scaling" and isinstance(
            baseline_val, (int, float)
        ):
            scale = item.get("scale_factor")
            if isinstance(scale, (int, float)):
                target_val = float(baseline_val) / float(scale)
        if evaluation.get("kind") == "close_cycle_capacity_model":
            target_reduction = item.get("target_reduction")
            if isinstance(target_reduction, (int, float)) and isinstance(
                baseline_val, (int, float)
            ):
                target_val = float(baseline_val) * (1.0 + float(target_reduction))
        if evaluation.get("kind") == "close_cycle_revenue_compression":
            target_val = item.get("target_days")

        status = evaluation.get("status")
        pass_fail_reason = "REVIEW"
        if status == "confirmed":
            pass_fail_reason = "PASS: expected evidence present."
        elif status == "missing":
            pass_fail_reason = "FAIL: no matching findings."
        elif status == "below_min":
            pass_fail_reason = "FAIL: observed below minimum threshold."
        elif status == "over_limit":
            pass_fail_reason = "FAIL: observed above maximum threshold."

        denominator = _denominator_text(item, report, spec)
        artifacts = _artifact_paths(report, evaluation.get("plugin_id"))
        impact_hours = _impact_hours(item)
        confidence_weight = _confidence_weight(item, evaluation.get("issue") or {})
        controllability_weight = _controllability_weight(
            evaluation.get("kind"), evaluation.get("issue") or {}
        )
        relevance_score = impact_hours * confidence_weight * controllability_weight

        cards.append(
            {
                "title": evaluation.get("label"),
                "plugin_id": evaluation.get("plugin_id"),
                "kind": evaluation.get("kind"),
                "metric_name": metric_name,
                "definition": definition,
                "denominator": denominator,
        "baseline": _format_issue_value(baseline_val),
        "target_threshold": _format_issue_value(target_val),
        "observed": _format_issue_value(observed_val),
                "status": status,
                "reason": pass_fail_reason,
                "artifact_paths": artifacts,
                "impact_hours": impact_hours,
                "confidence_weight": confidence_weight,
                "controllability_weight": controllability_weight,
                "relevance_score": relevance_score,
            }
        )
    return cards


def _waterfall_summary(report: dict[str, Any]) -> dict[str, Any] | None:
    plugins = report.get("plugins", {}) or {}
    queue_plugin = plugins.get("analysis_queue_delay_decomposition")
    if not isinstance(queue_plugin, dict):
        return None
    findings = queue_plugin.get("findings") or []
    impact = None
    qemail = None
    scale = None
    for item in findings:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        if kind == "eligible_wait_impact":
            impact = item
        elif kind == "capacity_scale_model":
            scale = item
        elif kind == "eligible_wait_process_stats" and item.get("process_norm") == "qemail":
            qemail = item
    if impact is None:
        return None
    if qemail is None:
        candidates = [
            f
            for f in findings
            if isinstance(f, dict) and f.get("kind") == "eligible_wait_process_stats"
        ]
        if candidates:
            qemail = max(
                candidates,
                key=lambda f: float(f.get("eligible_wait_gt_hours_total") or 0.0),
            )
    total = impact.get("eligible_wait_gt_hours_total")
    qemail_val = qemail.get("eligible_wait_gt_hours_total") if qemail else None
    remainder = impact.get("eligible_wait_gt_hours_without_target")
    modeled = scale.get("eligible_wait_gt_hours_modeled") if scale else None
    return {
        "total": total,
        "qemail": qemail_val,
        "remainder": remainder,
        "modeled": modeled,
        "scale_factor": scale.get("scale_factor") if scale else None,
    }


def _load_artifact_json(run_dir: Path, path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    target = run_dir / path
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _queue_delay_results(report: dict[str, Any], run_dir: Path) -> dict[str, Any] | None:
    plugin = report.get("plugins", {}).get("analysis_queue_delay_decomposition")
    if not isinstance(plugin, dict):
        return None
    artifacts = plugin.get("artifacts") or []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        path = artifact.get("path")
        if isinstance(path, str) and path.endswith("results.json"):
            payload = _load_artifact_json(run_dir, path)
            if payload:
                return payload
    return None


def _recommendation_merge_key(item: dict[str, Any]) -> tuple[str, str]:
    text = str(item.get("recommendation") or "").strip()
    title = str(item.get("title") or "").strip()
    plugin_id = str(item.get("plugin_id") or "")
    where = item.get("where") if isinstance(item.get("where"), dict) else None
    contains = item.get("contains") if isinstance(item.get("contains"), dict) else None
    scope_class = str(item.get("scope_class") or "").strip().lower()
    where_key = json_dumps(where) if where else ""
    contains_key = json_dumps(contains) if contains else ""
    delta = item.get("modeled_delta")
    delta_text = _format_issue_value(delta) if delta is not None else ""

    text_lower = text.lower()
    title_lower = title.lower()
    if text_lower.startswith("add one server") or "3rd server" in title_lower or "3rd server" in text_lower or "third server" in title_lower or "third server" in text_lower:
        base = "capacity_add_server"
    elif "close-cycle capacity" in title_lower or "close-cycle capacity" in text_lower:
        base = "close_cycle_capacity"
    elif text_lower.startswith("act on "):
        base = f"act_on:{plugin_id}:{where_key}:{contains_key}"
    elif text:
        base = text_lower
    else:
        base = title_lower
    merge_base = f"{scope_class}:{base}" if scope_class else base
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


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


def _collapse_findings(
    findings: list[Any], max_examples: int = 10
) -> dict[str, Any]:
    if not isinstance(findings, list) or not findings:
        return {"count": 0, "unique_count": 0, "top_examples": []}
    deduped: dict[str, dict[str, Any]] = {}
    for item in findings:
        key = json_dumps(item) if isinstance(item, (dict, list)) else str(item)
        if key not in deduped:
            deduped[key] = {"count": 0, "example": item}
        deduped[key]["count"] += 1
    ordered = sorted(
        deduped.values(), key=lambda entry: entry["count"], reverse=True
    )
    total = len(findings)
    return {
        "count": total,
        "unique_count": len(deduped),
        "top_examples": ordered[:max_examples],
    }


def _plugin_summary_rows(report: dict[str, Any]) -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]]]:
    rows: list[tuple[str, int, str]] = []
    plugins = report.get("plugins", {}) or {}
    for plugin_id, data in plugins.items():
        if not isinstance(data, dict):
            continue
        findings = data.get("findings") or []
        summary = (data.get("summary") or "").strip()
        rows.append((plugin_id, len(findings), summary))
    rows.sort()
    yes_rows = [row for row in rows if row[1] > 0]
    no_rows = [row for row in rows if row[1] == 0]
    return yes_rows, no_rows


def _format_plugin_table(rows: list[tuple[str, int, str]]) -> list[str]:
    lines = ["| Plugin | Findings | One-line summary |", "|---|---:|---|"]
    for plugin_id, count, summary in rows:
        lines.append(f"| `{plugin_id}` | {count} | {summary} |")
    return lines


def _load_known_issues_fallback(run_dir: Path) -> dict[str, Any] | None:
    known_dir = run_dir.parent.parent / "known_issues"
    if not known_dir.exists():
        return None
    payloads: list[dict[str, Any]] = []
    for path in sorted(known_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("expected_findings"):
            payloads.append(data)
    if not payloads:
        return None
    expected: list[dict[str, Any]] = []
    strict_values: list[bool] = []
    notes: list[str] = []
    for data in payloads:
        strict_values.append(bool(data.get("strict", False)))
        note = str(data.get("notes") or "").strip()
        if note:
            notes.append(note)
        expected.extend(data.get("expected_findings") or [])
    note_text = "Fallback merged from appdata/known_issues"
    if notes:
        note_text = f"{note_text} | " + " | ".join(notes)
    return {
        "scope_type": "fallback",
        "scope_value": "appdata/known_issues",
        "strict": all(strict_values) if strict_values else False,
        "notes": note_text,
        "natural_language": [],
        "expected_findings": expected,
    }

def build_report(
    storage: Storage, run_id: str, run_dir: Path, schema_path: Path
) -> dict[str, Any]:
    run_row = storage.fetch_run(run_id)
    if not run_row or not run_row.get("dataset_version_id"):
        raise ValueError("Run dataset version not found")
    upload_row = (
        storage.fetch_upload(run_row["upload_id"])
        if run_row.get("upload_id")
        else None
    )
    from .dataset_io import DatasetAccessor

    accessor = DatasetAccessor(storage, run_row["dataset_version_id"])
    info = accessor.info()

    plugin_rows = storage.fetch_plugin_results(run_id)
    plugins: dict[str, Any] = {}

    def _ensure_measurement(findings: list[Any]) -> list[Any]:
        allowed = {"measured", "modeled", "not_applicable", "error"}
        for item in findings:
            if not isinstance(item, dict):
                continue
            if "measurement_type" not in item:
                item["measurement_type"] = "measured"
            mt = item.get("measurement_type")
            if isinstance(mt, str):
                norm = mt.strip().lower()
                if norm == "degraded":
                    item["measurement_type"] = "not_applicable"
                elif norm in allowed:
                    item["measurement_type"] = norm
                else:
                    item["measurement_type"] = "error"
            else:
                item["measurement_type"] = "error"
        return findings

    def _canonicalize_payload(payload: Any) -> Any:
        try:
            return json.loads(json_dumps(payload))
        except TypeError:
            return payload

    def _sort_payload_list(items: list[Any]) -> list[Any]:
        try:
            return sorted(items, key=lambda item: json_dumps(item))
        except Exception:
            return items

    for row in sorted(plugin_rows, key=lambda item: item["plugin_id"]):
        findings = json.loads(row["findings_json"])
        if isinstance(findings, list):
            findings = _ensure_measurement(findings)
            findings = _sort_payload_list(findings)
        artifacts = json.loads(row["artifacts_json"])
        if isinstance(artifacts, list):
            artifacts = _sort_payload_list(artifacts)
        budget = None
        if "budget_json" in row.keys() and row.get("budget_json"):
            try:
                budget = json.loads(row["budget_json"])
            except json.JSONDecodeError:
                budget = None
        references = []
        if "references_json" in row.keys() and row.get("references_json"):
            try:
                references = json.loads(row["references_json"])
            except json.JSONDecodeError:
                references = []
        debug_info: dict[str, Any] = {}
        if "debug_json" in row.keys() and row.get("debug_json"):
            try:
                debug_info = json.loads(row["debug_json"])
            except json.JSONDecodeError:
                debug_info = {}
        if not isinstance(budget, dict):
            budget = {
                "row_limit": None,
                "sampled": False,
                "time_limit_ms": None,
                "cpu_limit_ms": None,
            }
        plugins[row["plugin_id"]] = {
            "status": row["status"],
            "summary": row["summary"],
            "metrics": _canonicalize_payload(json.loads(row["metrics_json"])),
            "findings": findings,
            "artifacts": artifacts,
            "budget": _canonicalize_payload(budget),
            "error": json.loads(row["error_json"]) if row["error_json"] else None,
            "references": _canonicalize_payload(references),
            "debug": _canonicalize_payload(debug_info),
        }

    dataset_version = storage.get_dataset_version(run_row["dataset_version_id"])
    dataset_context = storage.get_dataset_version_context(run_row["dataset_version_id"])
    project_row = None
    if dataset_context and dataset_context.get("project_id"):
        project_row = storage.fetch_project(dataset_context["project_id"])
    dataset_template = storage.fetch_dataset_template(run_row["dataset_version_id"])
    raw_format = None
    raw_format_id = None
    if dataset_version:
        raw_format_id = dataset_version.get("raw_format_id")
    if raw_format_id:
        with storage.connection() as conn:
            cur = conn.execute(
                """
                SELECT format_id, fingerprint, name, created_at
                FROM raw_formats
                WHERE format_id = ?
                """,
                (raw_format_id,),
            )
            row = cur.fetchone()
            raw_format = dict(row) if row else None
    if raw_format:
        raw_format = {
            "format_id": int(raw_format.get("format_id") or raw_format_id or 0),
            "fingerprint": raw_format.get("fingerprint") or "",
            "name": raw_format.get("name") or "",
            "created_at": raw_format.get("created_at") or "",
        }

    mapping = None
    if dataset_template and dataset_template.get("mapping_json"):
        try:
            mapping = json.loads(dataset_template["mapping_json"])
        except json.JSONDecodeError:
            mapping = {}

    dataset_block = {
        "dataset_version_id": run_row.get("dataset_version_id") or "unknown",
    }
    if dataset_context:
        if dataset_context.get("project_id"):
            dataset_block["project_id"] = dataset_context["project_id"]
        if dataset_context.get("dataset_id"):
            dataset_block["dataset_id"] = dataset_context["dataset_id"]
        if dataset_context.get("table_name"):
            dataset_block["table_name"] = dataset_context["table_name"]
    if dataset_version:
        if dataset_version.get("data_hash"):
            dataset_block["data_hash"] = dataset_version["data_hash"]
        if dataset_version.get("row_count") is not None:
            dataset_block["row_count"] = int(dataset_version["row_count"])
        if dataset_version.get("column_count") is not None:
            dataset_block["column_count"] = int(dataset_version["column_count"])
        if dataset_version.get("raw_format_id"):
            dataset_block["raw_format_id"] = int(dataset_version["raw_format_id"])

    def _string_or_empty(value: Any) -> str:
        return value if isinstance(value, str) else ""

    def _int_or_none(value: Any) -> int | None:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _float_or_none(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _rss_bytes_from_kb(value: Any) -> int | None:
        # On Linux, ru_maxrss is KB. We store KB in SQLite and derive bytes for reporting.
        kb = _int_or_none(value)
        if kb is None:
            return None
        if kb < 0:
            return None
        return int(kb) * 1024

    # Latest execution telemetry per plugin (duration/RSS/exit_code) for citeable + perf reporting.
    exec_rows = storage.fetch_plugin_executions(run_id)
    latest_exec_by_plugin: dict[str, dict[str, Any]] = {}
    for erow in exec_rows:
        pid = str(erow.get("plugin_id") or "")
        if not pid:
            continue
        prev = latest_exec_by_plugin.get(pid)
        if not prev or int(erow.get("execution_id") or 0) >= int(prev.get("execution_id") or 0):
            latest_exec_by_plugin[pid] = erow

    # Best-effort plugin type lookup (analysis/profile/transform/report/llm/etc) from plugin manifests.
    plugin_type_by_id: dict[str, str] = {}
    try:
        repo_root = Path(__file__).resolve().parents[3]
        plugins_root = repo_root / "plugins"
        for pid in latest_exec_by_plugin.keys():
            manifest = plugins_root / pid / "plugin.yaml"
            if not manifest.is_file():
                continue
            try:
                data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            except Exception:
                data = None
            if isinstance(data, dict):
                ptype = data.get("type")
                if isinstance(ptype, str) and ptype.strip():
                    plugin_type_by_id[pid] = ptype.strip()
    except Exception:
        plugin_type_by_id = {}

    lineage_plugins: dict[str, Any] = {}
    for row in plugin_rows:
        pid = str(row.get("plugin_id") or "")
        exec_row = latest_exec_by_plugin.get(pid) or {}
        lineage_plugins[pid] = {
            "plugin_version": _string_or_empty(row.get("plugin_version")),
            "code_hash": _string_or_empty(row.get("code_hash")),
            "settings_hash": _string_or_empty(row.get("settings_hash")),
            "dataset_hash": _string_or_empty(row.get("dataset_hash")),
            "executed_at": _string_or_empty(row.get("executed_at")),
            "status": _string_or_empty(row.get("status")),
            "summary": _string_or_empty(row.get("summary")),
            # Non-schema-critical additions (schema allows extra keys): execution + performance telemetry.
            "execution_fingerprint": _string_or_empty(row.get("execution_fingerprint")),
            "plugin_type": plugin_type_by_id.get(pid) or "",
            "execution": {
                "execution_id": _int_or_none(exec_row.get("execution_id")),
                "started_at": _string_or_empty(exec_row.get("started_at")),
                "completed_at": _string_or_empty(exec_row.get("completed_at")),
                "duration_ms": _int_or_none(exec_row.get("duration_ms")),
                "exit_code": _int_or_none(exec_row.get("exit_code")),
                "warnings_count": _int_or_none(exec_row.get("warnings_count")),
                "cpu_user": _float_or_none(exec_row.get("cpu_user")),
                "cpu_system": _float_or_none(exec_row.get("cpu_system")),
                "max_rss_kb": _int_or_none(exec_row.get("max_rss")),
                "max_rss_bytes": _rss_bytes_from_kb(exec_row.get("max_rss")),
            },
        }

    # Performance hotspots derived from execution telemetry. These are deterministic for a run_id.
    def _hotspot_row(erow: dict[str, Any]) -> dict[str, Any]:
        pid = str(erow.get("plugin_id") or "")
        return {
            "plugin_id": pid,
            "plugin_type": plugin_type_by_id.get(pid) or "",
            "status": str(erow.get("status") or ""),
            "duration_ms": _int_or_none(erow.get("duration_ms")),
            "exit_code": _int_or_none(erow.get("exit_code")),
            "max_rss_kb": _int_or_none(erow.get("max_rss")),
            "max_rss_bytes": _rss_bytes_from_kb(erow.get("max_rss")),
            "started_at": _string_or_empty(erow.get("started_at")),
            "completed_at": _string_or_empty(erow.get("completed_at")),
        }

    def _sort_key_duration(item: dict[str, Any]) -> tuple[int, str]:
        ms = _int_or_none(item.get("duration_ms"))
        return (-(ms or 0), str(item.get("plugin_id") or ""))

    def _sort_key_rss(item: dict[str, Any]) -> tuple[int, str]:
        kb = _int_or_none(item.get("max_rss_kb"))
        return (-(kb or 0), str(item.get("plugin_id") or ""))

    exec_payload = [_hotspot_row(r) for r in exec_rows if isinstance(r, dict)]
    top_duration = sorted(exec_payload, key=_sort_key_duration)[:15]
    top_rss = sorted(exec_payload, key=_sort_key_rss)[:15]
    failures = [
        item
        for item in sorted(
            exec_payload,
            key=lambda x: (str(x.get("status") or ""), str(x.get("plugin_id") or "")),
        )
        if (str(item.get("status") or "") not in {"ok", "na"})
        or (item.get("exit_code") not in {None, 0})
    ][:25]

    hotspots_block: dict[str, Any] = {
        "generated_at": now_iso(),
        "notes": "Telemetry sourced from plugin_executions (duration_ms + ru_maxrss in KB on Linux; bytes derived).",
        "top_by_duration_ms": top_duration,
        "top_by_max_rss_kb": top_rss,
        "failures": failures,
        "totals": {
            "plugin_executions": int(len(exec_payload)),
            "plugins_with_telemetry": int(len({p.get('plugin_id') for p in exec_payload if p.get('plugin_id')})),
        },
    }

    template_block = None
    if dataset_template:
        template_block = {
            "template_id": int(dataset_template["template_id"]),
            "table_name": dataset_template.get("table_name") or "",
            "status": dataset_template.get("status") or "",
            "mapping_hash": dataset_template.get("mapping_hash") or "",
            "mapping": mapping if isinstance(mapping, dict) else {},
        }
        if dataset_template.get("template_name"):
            template_block["template_name"] = dataset_template["template_name"]
        if dataset_template.get("template_version"):
            template_block["template_version"] = dataset_template["template_version"]

    known_payload = None
    if known_issues_enabled():
        known_block = None
        known_scope_type = ""
        known_scope_value = ""
        if project_row and project_row.get("erp_type"):
            known_scope_type = "erp_type"
            known_scope_value = str(project_row.get("erp_type") or "unknown").strip() or "unknown"
            known_block = storage.fetch_known_issues(known_scope_value, known_scope_type)
        if not known_block and upload_row and upload_row.get("sha256"):
            known_scope_type = "sha256"
            known_scope_value = str(upload_row.get("sha256") or "")
            if known_scope_value:
                known_block = storage.fetch_known_issues(known_scope_value, known_scope_type)
        if not known_block and dataset_block.get("data_hash"):
            data_hash = str(dataset_block.get("data_hash") or "")
            if re.fullmatch(r"[a-f0-9]{64}", data_hash):
                known_scope_type = "sha256"
                known_scope_value = data_hash
                known_block = storage.fetch_known_issues(known_scope_value, known_scope_type)

        if known_block:
            known_payload = {
                "scope_type": known_block.get("scope_type") or known_scope_type,
                "scope_value": known_block.get("scope_value") or known_scope_value,
                "strict": bool(known_block.get("strict", True)),
                "notes": known_block.get("notes") or "",
                "natural_language": known_block.get("natural_language") or [],
                "expected_findings": known_block.get("expected_findings") or [],
            }
        if project_row and str(project_row.get("erp_type") or "").strip().lower() == "quorum":
            if not known_payload:
                known_payload = {
                    "scope_type": "erp_type",
                    "scope_value": "quorum",
                    "strict": False,
                    "notes": "",
                    "natural_language": [],
                    "expected_findings": [],
                }
            exclusions = known_payload.get("recommendation_exclusions")
            if not isinstance(exclusions, dict):
                exclusions = {}
            processes = exclusions.get("processes")
            if not isinstance(processes, list):
                processes = []
            quorum_exclusions = {"postwkfl", "bkrvnu", "cwowfndrls"}
            merged = sorted({*(str(p).strip() for p in processes if str(p).strip()), *quorum_exclusions})
            exclusions["processes"] = merged
            known_payload["recommendation_exclusions"] = exclusions

    orchestrator_mode = "two_lane_strict"
    try:
        settings_raw = run_row.get("settings_json")
        settings_payload = json.loads(settings_raw) if isinstance(settings_raw, str) and settings_raw.strip() else {}
        if isinstance(settings_payload, dict):
            sys_payload = settings_payload.get("_system")
            if isinstance(sys_payload, dict):
                mode_raw = str(sys_payload.get("orchestrator_mode") or "").strip().lower()
                if mode_raw:
                    orchestrator_mode = mode_raw
    except Exception:
        orchestrator_mode = "two_lane_strict"
    overall_outcome = "passed" if str(run_row.get("status") or "").strip().lower() == "completed" else "failed"

    report = {
        "run_id": run_id,
        "created_at": now_iso(),
        "status": run_row.get("status") or "completed",
        "run_fingerprint": run_row.get("run_fingerprint") or "",
        "known_issues_mode": known_issues_mode_label(),
        "orchestrator_mode": orchestrator_mode,
        "hotspots": hotspots_block,
        "input": {
            "filename": run_row.get("input_filename") or "unknown",
            **info,
        },
        "lineage": {
            "run": {
                "run_id": run_id,
                "created_at": run_row.get("created_at") or "",
                "status": run_row.get("status") or "",
                "overall_outcome": overall_outcome,
                "orchestrator_mode": orchestrator_mode,
                "run_seed": int(run_row.get("run_seed") or 0),
                "run_fingerprint": run_row.get("run_fingerprint") or "",
            },
            "input": {
                "upload_id": run_row.get("upload_id") or "",
                "filename": run_row.get("input_filename") or "unknown",
                "canonical_path": run_row.get("canonical_path") or "",
                "input_hash": run_row.get("input_hash") or "",
                "sha256": upload_row.get("sha256") if upload_row else "",
                "size_bytes": int(upload_row.get("size_bytes") or 0)
                if upload_row
                else 0,
            },
            "dataset": dataset_block,
            "raw_format": raw_format,
            "template": template_block,
            "plugins": lineage_plugins,
        },
        "plugins": plugins,
    }
    if known_issues_enabled() and not known_payload:
        known_payload = _load_known_issues_fallback(run_dir)
    if known_payload:
        report["known_issues"] = known_payload

    # Persist a lightweight "ideaspace" index so repeated ERP-like datasets can be
    # compared and so recommendations can be traced back to columns + insight families.
    erp_type = ""
    if project_row and project_row.get("erp_type"):
        erp_type = str(project_row.get("erp_type") or "").strip()
    if not erp_type and isinstance(known_payload, dict) and known_payload.get("scope_type") == "erp_type":
        erp_type = str(known_payload.get("scope_value") or "").strip()

    columns_index: list[dict[str, Any]] = []
    try:
        dataset_version_id = run_row["dataset_version_id"]
        ds_template = storage.fetch_dataset_template(dataset_version_id)
        if ds_template and ds_template.get("status") == "ready":
            fields = storage.fetch_template_fields(int(ds_template["template_id"]))
            for field in fields:
                columns_index.append(
                    {
                        "name": field.get("name") or "",
                        "dtype": field.get("dtype"),
                        "role": field.get("role") or "",
                    }
                )
        else:
            cols = storage.fetch_dataset_columns(dataset_version_id)
            for col in cols:
                columns_index.append(
                    {
                        "name": col.get("original_name") or "",
                        "dtype": col.get("dtype"),
                        "role": col.get("role") or "",
                    }
                )
    except Exception:
        columns_index = []

    insight_index: list[dict[str, Any]] = []
    for plugin_id, plugin in plugins.items():
        if not isinstance(plugin, dict):
            continue
        kinds: dict[str, int] = {}
        for item in plugin.get("findings") or []:
            if not isinstance(item, dict):
                continue
            kind = item.get("kind")
            if isinstance(kind, str) and kind:
                kinds[kind] = kinds.get(kind, 0) + 1
        if kinds:
            insight_index.append(
                {
                    "plugin_id": plugin_id,
                    "status": plugin.get("status"),
                    "kinds": kinds,
                }
            )

    landmarks: list[dict[str, Any]] = []
    if isinstance(known_payload, dict):
        for exp in known_payload.get("expected_findings") or []:
            if not isinstance(exp, dict):
                continue
            landmarks.append(
                {
                    "title": exp.get("title") or "",
                    "plugin_id": exp.get("plugin_id") or "",
                    "kind": exp.get("kind") or "",
                    "where": exp.get("where") if isinstance(exp.get("where"), dict) else None,
                    "contains": exp.get("contains") if isinstance(exp.get("contains"), dict) else None,
                }
            )

    report["ideaspace"] = {
        "erp_type": erp_type or None,
        "columns": columns_index,
        "insight_index": insight_index,
        "landmarks": landmarks,
        # Sprint 7.3: expand the ideaspace index with role detection + applicability.
        "roles": _infer_ideaspace_roles(columns_index),
        "normalization": {
            "process_norm": "lowercase",
            "notes": "process_norm values are normalized via `.strip().lower()` unless provided by a template.",
        },
        "families": _ideaspace_families_summary(plugins),
    }

    report["recommendations"] = _build_recommendations(report, storage, run_dir=run_dir)
    report["four_pillars"] = build_four_pillars_scorecard(report, run_row=run_row)
    evaluation_path = run_dir / "evaluation.json"
    if evaluation_path.exists():
        try:
            report["evaluation"] = json.loads(evaluation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report["evaluation"] = None

    schema = read_json(schema_path)
    validate(instance=report, schema=schema)
    return report


def write_report(report: dict[str, Any], run_dir: Path) -> None:
    report_path = run_dir / "report.json"
    write_json(report_path, report)

    lines = ["# Statistic Harness Report", ""]
    # Sprint 7.2: CEO-grade "Top Actions" section derived from Family F plugin.
    ops = (report.get("plugins") or {}).get("analysis_actionable_ops_levers_v1")
    actions = ops.get("findings") if isinstance(ops, dict) else None
    if isinstance(actions, list) and actions:
        lines.append("## Top Actions")
        lines.append("")
        shown = 0
        for item in actions:
            if not isinstance(item, dict):
                continue
            if item.get("kind") != "actionable_ops_lever":
                continue
            proc = item.get("process_norm") or item.get("process") or "process"
            action_type = item.get("action_type") or "action"
            delta_s = item.get("expected_delta_seconds")
            delta_pct = item.get("expected_delta_percent")
            conf = item.get("confidence")
            parts = [f"**{proc}**", f"`{action_type}`"]
            if isinstance(delta_s, (int, float)):
                parts.append(f"Δ {float(delta_s):.2f}s")
            if isinstance(delta_pct, (int, float)):
                parts.append(f"({float(delta_pct):.1f}%)")
            if isinstance(conf, (int, float)):
                parts.append(f"conf={float(conf):.2f}")
            from_node = item.get("from")
            to_node = item.get("to")
            if isinstance(from_node, str) and isinstance(to_node, str) and from_node and to_node:
                parts.append(f"{from_node} -> {to_node}")
            lines.append(f"- {' '.join(parts)}")
            shown += 1
            if shown >= 20:
                break
        lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"Run ID: {report.get('run_id')}")
    lines.append(f"Created: {report.get('created_at')}")
    lines.append(f"Status: {report.get('status')}")
    lines.append(f"Rows: {report.get('input', {}).get('rows')}")
    lines.append(f"Cols: {report.get('input', {}).get('cols')}")
    lines.append("")

    four = report.get("four_pillars")
    if isinstance(four, dict):
        lines.append("### 4-Pillar Scorecard")
        lines.append("")
        summary = four.get("summary") if isinstance(four.get("summary"), dict) else {}
        balance = four.get("balance") if isinstance(four.get("balance"), dict) else {}
        overall = summary.get("overall_0_4")
        status = summary.get("status")
        lines.append(
            f"Overall (balanced): {float(overall):.2f}/4.00"
            if isinstance(overall, (int, float))
            else "Overall (balanced): n/a"
        )
        if isinstance(status, str) and status:
            lines.append(f"Status: {status}")
        spread = balance.get("spread")
        min_p = balance.get("min_pillar")
        max_p = balance.get("max_pillar")
        if isinstance(min_p, (int, float)) and isinstance(max_p, (int, float)):
            lines.append(
                f"Balance: min={float(min_p):.2f}, max={float(max_p):.2f}, spread={float(spread or 0.0):.2f}"
            )
        pillars = four.get("pillars")
        if isinstance(pillars, dict):
            for pillar in ("performant", "accurate", "secure", "citable"):
                payload = pillars.get(pillar)
                if not isinstance(payload, dict):
                    continue
                score = payload.get("score_0_4")
                if isinstance(score, (int, float)):
                    lines.append(f"- {pillar}: {float(score):.2f}/4.00")
        vetoes = balance.get("vetoes")
        if isinstance(vetoes, list) and vetoes:
            lines.append("Vetoes:")
            for item in vetoes:
                if not isinstance(item, dict):
                    continue
                code = str(item.get("code") or "unknown")
                message = str(item.get("message") or "").strip()
                lines.append(f"- `{code}`: {message}" if message else f"- `{code}`")
        lines.append("")

    exec_summary = _build_executive_summary(report)
    lines.append("### Executive Summary")
    if exec_summary:
        for entry in exec_summary:
            lines.append(f"- {entry}")
    else:
        lines.append("No executive summary available.")
    lines.append("")

    cards = _issue_cards(report)
    cards_sorted = sorted(cards, key=lambda c: c.get("relevance_score", 0.0), reverse=True)
    top_n = int(os.environ.get("STAT_HARNESS_DECISION_TOP_N", "5"))
    decision_cards = cards_sorted[:top_n] if top_n > 0 else []

    lines.append("### Decision Items")
    if decision_cards:
        lines.append("| Issue | Score | Status | Reason |")
        lines.append("|---|---:|---|---|")
        for card in decision_cards:
            title = card.get("title") or "Issue"
            score = _format_issue_value(card.get("relevance_score"), digits=3)
            status = card.get("status") or "unknown"
            reason = card.get("reason") or ""
            lines.append(f"| {title} | {score} | {status} | {reason} |")
    else:
        lines.append("No decision items available.")
    lines.append("")

    lines.append("### Waterfall Summary")
    waterfall = _waterfall_summary(report)
    if waterfall and all(
        isinstance(waterfall.get(key), (int, float)) for key in ("total", "qemail", "remainder")
    ):
        total = float(waterfall["total"])
        qemail = float(waterfall["qemail"])
        remainder = float(waterfall["remainder"])
        modeled = waterfall.get("modeled")
        scale = waterfall.get("scale_factor")
        total_text = _format_issue_value(total)
        qemail_text = _format_issue_value(qemail)
        remainder_text = _format_issue_value(remainder)
        lines.append(f"Total over-threshold eligible wait: {total_text}h")
        lines.append(f"QEMAIL contribution: {qemail_text}h")
        lines.append(
            f"Remainder without QEMAIL: {remainder_text}h = {total_text}h - {qemail_text}h"
        )
        if isinstance(modeled, (int, float)):
            modeled_text = _format_issue_value(float(modeled))
            if isinstance(scale, (int, float)):
                scale_text = _format_issue_value(float(scale))
                lines.append(
                    f"Modeled after add one server: {modeled_text}h = {remainder_text}h × {scale_text}"
                )
            else:
                lines.append(f"Modeled after add one server: {modeled_text}h")
    else:
        lines.append("No waterfall summary available.")
    lines.append("")

    recommendations = report.get("recommendations")
    lines.append("### Recommendations")
    known_items, discovery_items, known_summary, discovery_summary = _split_recommendations(
        recommendations if isinstance(recommendations, dict) else None
    )
    excluded_processes = _explicit_excluded_processes(report)

    lines.append("#### Discovery Recommendations")
    if discovery_summary:
        lines.append(discovery_summary)
    if discovery_items:
        discovery_items = _dedupe_recommendations_text(discovery_items)
        discovery_items = _filter_recommendations_by_process(
            discovery_items, excluded_processes
        )
        close_items = [
            item
            for item in discovery_items
            if str(item.get("scope_class") or "").strip().lower() == "close_specific"
        ]
        general_items = [
            item
            for item in discovery_items
            if str(item.get("scope_class") or "").strip().lower() != "close_specific"
        ]
        for label, scoped_items in (
            ("Close-Specific", close_items),
            ("General", general_items),
        ):
            lines.append(f"##### {label}")
            if not scoped_items:
                lines.append("No recommendations in this scope.")
                continue
            lines.append("| Status | Recommendation |")
            lines.append("|---|---|")
            for item in scoped_items:
                status = item.get("status") or "unknown"
                rec = item.get("recommendation") or "Recommendation"
                modeled_pct = item.get("modeled_percent")
                if isinstance(modeled_pct, (int, float)):
                    rec = f"{rec} (modeled={float(modeled_pct):.2f}%)"
                merged_titles = [
                    title
                    for title in (item.get("merged_titles") or [])
                    if isinstance(title, str) and title
                ]
                if len(merged_titles) > 1:
                    merged_text = "; ".join(dict.fromkeys(merged_titles))
                    rec = f"{rec} (covers: {merged_text})"
                lines.append(f"| {status} | {rec} |")
    else:
        lines.append("No discovery recommendations available.")
    lines.append("")

    explanations_block = (
        recommendations.get("explanations")
        if isinstance(recommendations, dict)
        else None
    )
    explanation_items = (
        explanations_block.get("items")
        if isinstance(explanations_block, dict)
        and isinstance(explanations_block.get("items"), list)
        else []
    )
    lines.append("#### Non-Actionable Explanations")
    if isinstance(explanations_block, dict):
        summary_text = str(explanations_block.get("summary") or "").strip()
        if summary_text:
            lines.append(summary_text)
    if explanation_items:
        lines.append("| Plugin | Reason | Explanation |")
        lines.append("|---|---|---|")
        for item in explanation_items:
            if not isinstance(item, dict):
                continue
            plugin_id = str(item.get("plugin_id") or "unknown")
            reason = str(item.get("reason_code") or "unspecified")
            text = str(item.get("plain_english_explanation") or "").strip() or "No explanation provided."
            downstream = item.get("downstream_plugins")
            if isinstance(downstream, list) and downstream:
                sample = ", ".join(str(v) for v in downstream[:5])
                suffix = ", ..." if len(downstream) > 5 else ""
                text = f"{text} Downstream: {sample}{suffix}."
            lines.append(f"| {plugin_id} | {reason} | {text} |")
    else:
        lines.append("No non-actionable explanations available.")
    lines.append("")

    if _include_known_recommendations():
        lines.append("#### Known-Issue Recommendations (Pass Gate)")
        if known_summary:
            lines.append(known_summary)
        if known_items:
            known_items = _dedupe_recommendations_text(known_items)
            lines.append("| Status | Recommendation |")
            lines.append("|---|---|")
            for item in known_items:
                status = item.get("status") or "unknown"
                rec = item.get("recommendation") or "Recommendation"
                merged_titles = [
                    title
                    for title in (item.get("merged_titles") or [])
                    if isinstance(title, str) and title
                ]
                if len(merged_titles) > 1:
                    merged_text = "; ".join(dict.fromkeys(merged_titles))
                    rec = f"{rec} (covers: {merged_text})"
                lines.append(f"| {status} | {rec} |")
        else:
            lines.append("No known-issue recommendations available.")
        lines.append("")
    else:
        lines.append(
            "Known-issue landmarks are excluded from recommendations by policy; see Decision Items for gate status."
        )
        lines.append("")

    if decision_cards:
        lines.append("### Issue Cards (Top)")
        for card in decision_cards:
            lines.append(f"#### {card.get('title')}")
            lines.append(
                f"Metric: {card.get('metric_name')} — {card.get('definition')}"
            )
            lines.append(f"Denominator: {card.get('denominator')}")
            lines.append(f"Baseline: {card.get('baseline')}")
            lines.append(f"Target Threshold: {card.get('target_threshold')}")
            lines.append(f"Observed: {card.get('observed')}")
            lines.append(f"Decision: {card.get('reason')}")
            artifacts = ", ".join(card.get("artifact_paths") or []) or "n/a"
            lines.append(f"Evidence Artifacts: {artifacts}")
            lines.append("")

    lines.append("## Appendix")
    lines.append("")
    lines.append("### Issue Cards (Full)")
    if cards:
        for card in cards:
            lines.append(f"#### {card.get('title')}")
            lines.append(
                f"Metric: {card.get('metric_name')} — {card.get('definition')}"
            )
            lines.append(f"Denominator: {card.get('denominator')}")
            lines.append(f"Baseline: {card.get('baseline')}")
            lines.append(f"Target Threshold: {card.get('target_threshold')}")
            lines.append(f"Observed: {card.get('observed')}")
            lines.append(f"Decision: {card.get('reason')}")
            artifacts = ", ".join(card.get("artifact_paths") or []) or "n/a"
            lines.append(f"Evidence Artifacts: {artifacts}")
            lines.append("")
    else:
        lines.append("No issue cards available.")
        lines.append("")

    lines.append("### Plugin Dumps")
    for plugin_id in sorted(report.get("plugins", {}).keys()):
        data = _trim_plugin_dump_for_markdown(report["plugins"][plugin_id])
        lines.append(f"#### {plugin_id}")
        lines.append("```json")
        lines.append(json_dumps(data))
        lines.append("```")
        lines.append("")
    atomic_write_text(run_dir / "report.md", "\n".join(lines) + "\n")

    # Avoid clobbering plugin-emitted artifacts.
    # Report-stage plugins (for example report_decision_bundle_v2 and report_slide_kit_emitter_v2)
    # run before report_bundle and write these same files. Keep their output as source-of-truth.
    if not (run_dir / "business_summary.md").exists():
        _write_business_summary(report, run_dir)
    if not (run_dir / "engineering_summary.md").exists():
        _write_engineering_summary(report, run_dir)
    if not (run_dir / "appendix_raw.md").exists():
        _write_appendix_raw(report, run_dir)

    kit_dir = run_dir / "slide_kit"
    if not (
        (kit_dir / "scenario_summary.csv").exists()
        or (kit_dir / "top_process_contributors.csv").exists()
        or (kit_dir / "busy_periods.csv").exists()
    ):
        _write_slide_kit(report, run_dir)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (int, bool)):
        return str(value)
    if isinstance(value, list):
        preview = ", ".join(str(v) for v in value[:5])
        if len(value) > 5:
            preview += ", ..."
        return f"[{preview}]"
    if isinstance(value, dict):
        return "{...}"
    text = str(value)
    if len(text) > 80:
        return text[:77] + "..."
    return text


def _format_metrics(metrics: dict[str, Any]) -> list[str]:
    if not isinstance(metrics, dict):
        return []
    items: list[tuple[str, Any]] = []
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)):
            items.append((key, value))
    items = sorted(items, key=lambda item: item[0])
    return [f"{key}: {_format_value(value)}" for key, value in items]


def _format_findings(findings: list[Any]) -> list[str]:
    rendered: list[str] = []
    for finding in findings:
        if not isinstance(finding, dict):
            rendered.append(_format_value(finding))
            continue
        kind = finding.get("kind", "finding")
        measurement = finding.get("measurement_type", "measured")
        parts = [f"kind={kind}", f"measurement={measurement}"]
        key_fields = [
            "role",
            "column",
            "process",
            "process_norm",
            "process_id",
            "process_name",
            "module",
            "module_cd",
            "user",
            "user_id",
            "dimension",
            "key",
            "sequence",
            "host",
            "feature",
            "metric",
        ]
        for field in key_fields:
            if field in finding and finding[field] not in (None, ""):
                parts.append(f"{field}={_format_value(finding[field])}")
        numeric_fields = [
            key
            for key, value in finding.items()
            if isinstance(value, (int, float))
            and key not in {"row_index"}
            and (
                key.endswith("_sec")
                or key.endswith("_hours")
                or key.endswith("_count")
                or key.endswith("_runs")
                or key.endswith("_ratio")
                or key.endswith("_pct")
                or key in {"p50", "p95", "p99", "mean", "min", "max", "score"}
            )
        ]
        for key in sorted(numeric_fields)[:6]:
            parts.append(f"{key}={_format_value(finding[key])}")
        evidence = finding.get("evidence")
        if isinstance(evidence, dict):
            row_ids = evidence.get("row_ids")
            col_ids = evidence.get("column_ids")
            if isinstance(row_ids, list) and row_ids:
                parts.append(f"rows={len(row_ids)}")
            if isinstance(col_ids, list) and col_ids:
                parts.append(f"cols={len(col_ids)}")
            query = evidence.get("query")
            if query:
                parts.append(f"query={_format_value(query)}")
        rendered.append(", ".join(parts))
    return rendered


def _matches_expected(
    item: dict[str, Any], where: dict[str, Any] | None, contains: dict[str, Any] | None
) -> bool:
    if where:
        for key, expected in where.items():
            if item.get(key) != expected:
                return False
    if contains:
        for key, expected in contains.items():
            actual = item.get(key)
            if isinstance(actual, str):
                if str(expected) not in actual:
                    return False
            elif isinstance(actual, (list, tuple, set)):
                if isinstance(expected, (list, tuple, set)):
                    if not set(expected).issubset(set(actual)):
                        return False
                else:
                    if expected not in actual:
                        return False
            else:
                return False
    return True


def _format_known_issue_checks(
    report: dict[str, Any], expected: list[dict[str, Any]]
) -> list[str]:
    items: list[str] = []
    for entry in expected:
        if not isinstance(entry, dict):
            continue
        plugin_id = entry.get("plugin_id")
        kind = entry.get("kind")
        if not kind:
            continue
        where = entry.get("where") or {}
        contains = entry.get("contains") or {}
        min_count = int(entry.get("min_count", 1))
        max_count = entry.get("max_count")
        candidates = []
        for pid, plugin in report.get("plugins", {}).items():
            if plugin_id and pid != plugin_id:
                continue
            for item in plugin.get("findings", []):
                if item.get("kind") == kind:
                    candidates.append(item)
        matches = [item for item in candidates if _matches_expected(item, where, contains)]
        status = "PASS" if len(matches) >= min_count and (max_count is None or len(matches) <= int(max_count)) else "FAIL"
        detail = f"{len(matches)} match(es)"
        title = entry.get("title") or entry.get("description") or ""
        if title:
            title = title.strip()
        context = f"{kind} ({plugin_id or '*'})"
        if title:
            context = f"{title} :: {context}"
        items.append(f"{status} - {context} - {detail}")
    return items


def _metric_unit(metric_name: str, item: dict[str, Any] | None = None) -> str:
    name = metric_name.lower()
    if "%" in name or "percent" in name or "pct" in name:
        return "percent"
    if "hours" in name:
        return "hours"
    if "seconds" in name or "sec" in name:
        return "seconds"
    if "days" in name:
        return "days"
    if item:
        for key in item.keys():
            if key.endswith("_hours"):
                return "hours"
            if key.endswith("_sec"):
                return "seconds"
            if key.endswith("_days"):
                return "days"
            if key.endswith("_pct"):
                return "percent"
    return "count"


def _scenario_rows_from_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(item, dict):
        return []
    baseline = None
    modeled = None
    unit = None
    if "eligible_wait_gt_hours_without_target" in item and "eligible_wait_gt_hours_modeled" in item:
        baseline = item.get("eligible_wait_gt_hours_without_target")
        modeled = item.get("eligible_wait_gt_hours_modeled")
        unit = "hours"
    elif "baseline_wait_hours" in item and "modeled_wait_hours" in item:
        baseline = item.get("baseline_wait_hours")
        modeled = item.get("modeled_wait_hours")
        unit = "hours"
    elif "baseline_median_sec" in item and "modeled_median_sec" in item:
        baseline = item.get("baseline_median_sec")
        modeled = item.get("modeled_median_sec")
        unit = "seconds"
    elif "baseline_span_days_median" in item and "target_days" in item:
        baseline = item.get("baseline_span_days_median")
        modeled = item.get("target_days")
        unit = "days"

    if not isinstance(baseline, (int, float)) or not isinstance(modeled, (int, float)):
        return []
    delta = float(modeled) - float(baseline)
    return [
        {"scenario": "current", "value": float(baseline), "delta": 0.0, "unit": unit},
        {"scenario": "modeled", "value": float(modeled), "delta": delta, "unit": unit},
    ]


def _busy_periods_info(report: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    results = _queue_delay_results(report, run_dir)
    if not results:
        return {
            "busy_periods": [],
            "threshold_seconds": None,
            "bucket": None,
            "basis": None,
            "artifact_path": None,
        }
    summary = results.get("summary") or {}
    busy_periods = results.get("busy_periods") or []
    artifact_path = None
    plugin = report.get("plugins", {}).get("analysis_queue_delay_decomposition") or {}
    for artifact in plugin.get("artifacts", []) or []:
        if isinstance(artifact, dict) and str(artifact.get("path", "")).endswith("results.json"):
            artifact_path = artifact.get("path")
            break
    return {
        "busy_periods": busy_periods,
        "threshold_seconds": summary.get("wait_threshold_seconds"),
        "bucket": summary.get("busy_period_bucket"),
        "basis": summary.get("busy_period_basis"),
        "artifact_path": artifact_path,
        "summary": summary,
    }


def _top_process_contributors(report: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
    plugin = report.get("plugins", {}).get("analysis_queue_delay_decomposition")
    if not isinstance(plugin, dict):
        return []
    findings = plugin.get("findings") or []
    process_stats = [
        f for f in findings if isinstance(f, dict) and f.get("kind") == "eligible_wait_process_stats"
    ]
    impact = next(
        (f for f in findings if isinstance(f, dict) and f.get("kind") == "eligible_wait_impact"),
        None,
    )
    total = impact.get("eligible_wait_gt_hours_total") if isinstance(impact, dict) else None
    rows: list[dict[str, Any]] = []
    for item in process_stats:
        value = item.get("eligible_wait_gt_hours_total")
        if not isinstance(value, (int, float)):
            continue
        share = float(value) / float(total) if isinstance(total, (int, float)) and total else None
        rows.append(
            {
                "process": item.get("process") or item.get("process_norm"),
                "eligible_wait_gt_hours_total": float(value),
                "runs_total": int(item.get("runs_total") or 0),
                "share_of_total": share,
            }
        )
    rows.sort(key=lambda r: r["eligible_wait_gt_hours_total"], reverse=True)
    return rows[:limit]


def _qemail_profile(report: dict[str, Any]) -> dict[str, Any] | None:
    plugin = report.get("plugins", {}).get("analysis_queue_delay_decomposition")
    if not isinstance(plugin, dict):
        return None
    findings = plugin.get("findings") or []
    for item in findings:
        if (
            isinstance(item, dict)
            and item.get("kind") == "eligible_wait_process_stats"
            and item.get("process_norm") == "qemail"
        ):
            return {
                "process": item.get("process"),
                "eligible_wait_gt_hours_total": item.get("eligible_wait_gt_hours_total"),
                "eligible_wait_hours_total": item.get("eligible_wait_hours_total"),
                "runs_total": item.get("runs_total"),
                "runs_close": item.get("runs_close"),
                "runs_open": item.get("runs_open"),
            }
    return None


def _write_business_summary(report: dict[str, Any], run_dir: Path) -> None:
    info = _busy_periods_info(report, run_dir)
    busy_periods = info.get("busy_periods") or []
    threshold = info.get("threshold_seconds")
    summary = info.get("summary") or {}
    close_start = summary.get("close_cycle_start_day")
    close_end = summary.get("close_cycle_end_day")
    artifact_path = info.get("artifact_path") or "n/a"
    busy_periods_sorted = sorted(
        busy_periods,
        key=lambda b: float(b.get("wait_to_start_hours_total") or 0.0),
        reverse=True,
    )
    top_busy = busy_periods_sorted[:10]
    total_busy_hours = sum(
        float(item.get("wait_to_start_hours_total") or 0.0) for item in busy_periods_sorted
    )

    lines = ["# Business Summary", ""]
    threshold_text = f"{int(threshold)}s" if isinstance(threshold, (int, float)) else "n/a"
    if isinstance(close_start, int) and isinstance(close_end, int):
        close_window = f"days {close_start}-{close_end} of month"
    else:
        close_window = "all days"
    lines.append(
        "Primary KPI: busy-period wait-to-start hours, defined as total wait-to-start hours "
        "in hourly buckets (by queue timestamp) within the close-cycle window "
        f"({close_window}) where wait-to-start exceeds {threshold_text}."
    )
    lines.append(
        "Metric context: unit=hours; population=standalone, non-excluded rows in close-cycle window "
        "with queue/start timestamps; "
        f"threshold=wait-to-start > {threshold_text}; filters=close_cycle=true; busy_period_bucket=hour."
    )
    lines.append("")

    lines.append("## Busy Periods (Top 10)")
    if top_busy:
        lines.append(
            "Metric context: unit=hours; population=standalone, non-excluded rows in close-cycle window "
            f"with queue/start timestamps; threshold=wait-to-start > {threshold_text}; "
            "filters=close_cycle=true; busy_period_bucket=hour."
        )
        lines.append("| Period Start | Period End | Wait-to-Start Hours | Rows | After-hours | Weekend |")
        lines.append("|---|---|---:|---:|---|---|")
        for row in top_busy:
            lines.append(
                "| {start} | {end} | {hours:.3f} | {rows} | {after} | {weekend} |".format(
                    start=row.get("period_start"),
                    end=row.get("period_end"),
                    hours=float(row.get("wait_to_start_hours_total") or 0.0),
                    rows=int(row.get("rows_total") or 0),
                    after="yes" if row.get("after_hours") else "no",
                    weekend="yes" if row.get("weekend") else "no",
                )
            )
        weekday_count = sum(1 for row in top_busy if not row.get("weekend"))
        after_hours_count = sum(1 for row in top_busy if row.get("after_hours"))
        lines.append(
            f"Note: {weekday_count}/{len(top_busy)} periods are weekdays; "
            f"{after_hours_count}/{len(top_busy)} occur after-hours."
        )
        lines.append(f"Full detail: {artifact_path}")
    else:
        lines.append("No busy period data available.")
    lines.append("")

    known_items, discovery_items, known_summary, discovery_summary = _split_recommendations(
        report.get("recommendations") if isinstance(report.get("recommendations"), dict) else None
    )
    excluded_processes = _explicit_excluded_processes(report)
    excluded_processes = _explicit_excluded_processes(report)
    discovery_items = _dedupe_recommendations_text(discovery_items)
    discovery_items = _filter_recommendations_by_process(
        discovery_items, excluded_processes
    )
    discovery_items = _filter_recommendations_by_process(
        discovery_items, excluded_processes
    )
    recs_sorted = sorted(discovery_items, key=_final_recommendation_sort_key)[:3]

    if recs_sorted:
        lines.append("## Top Recommendations (Discovery)")
        if discovery_summary:
            lines.append(discovery_summary)
    for idx, rec in enumerate(recs_sorted, start=1):
        title = rec.get("title") or f"Recommendation {idx}"
        merged_titles = [
            text
            for text in (rec.get("merged_titles") or [])
            if isinstance(text, str) and text
        ]
        metric_name = rec.get("metric_name") or "Metric"
        unit = rec.get("unit") or _metric_unit(metric_name, rec)
        denominator = rec.get("denominator") or "n/a"
        filter_text = ""
        if rec.get("where"):
            filter_text = f"where={rec.get('where')}"
        if rec.get("contains"):
            filter_text = f"{filter_text} contains={rec.get('contains')}".strip()
        if not filter_text:
            filter_text = f"threshold={threshold_text}"
        measurement_type = rec.get("measurement_type") or "measured"
        confidence_tag = "Modeled" if measurement_type == "modeled" else "Measured"

        lines.append(f"### {title}")
        lines.append(f"Problem: {title}")
        if len(merged_titles) > 1:
            merged_text = "; ".join(dict.fromkeys(merged_titles))
            lines.append(f"Also covers: {merged_text}")
        baseline = rec.get("baseline")
        observed = rec.get("observed")
        target = rec.get("target_threshold")
        if baseline is not None or observed is not None or target is not None:
            lines.append(
                "Evidence: "
                f"baseline={_format_issue_value(baseline)}, observed={_format_issue_value(observed)}, "
                f"target={_format_issue_value(target)}."
            )
        else:
            evidence = rec.get("evidence") or []
            numbers = evidence[0] if evidence else {}
            if isinstance(numbers, dict) and numbers:
                detail = ", ".join(
                    f"{key}={_format_issue_value(numbers.get(key))}"
                    for key in list(numbers.keys())[:4]
                )
                lines.append(f"Evidence: {detail}.")
        lines.append(f"Action: {rec.get('recommendation')}")
        lines.append(
            f"Metric context: unit={unit}; population={denominator}; threshold/filters={filter_text}."
        )
        scenario_rows = []
        if rec.get("kind") == "upload_bkrvnu_linkage":
            current_val = rec.get("observed")
            if not isinstance(current_val, (int, float)):
                current_val = rec.get("baseline")
            if isinstance(current_val, (int, float)):
                scenario_rows.append(
                    {
                        "scenario": "current",
                        "value": float(current_val) * 100.0,
                        "delta": 0.0,
                        "unit": "percent",
                    }
                )
                target_val = 1.0
                scenario_rows.append(
                    {
                        "scenario": "target_linked",
                        "value": float(target_val) * 100.0,
                        "delta": (float(target_val) - float(current_val)) * 100.0,
                        "unit": "percent",
                    }
                )
        else:
            impact_hours = rec.get("impact_hours")
            scenario_rows.append(
                {
                    "scenario": "current",
                    "value": float(total_busy_hours),
                    "delta": 0.0,
                    "unit": "hours",
                }
            )
            if isinstance(impact_hours, (int, float)) and float(total_busy_hours):
                modeled_value = max(0.0, float(total_busy_hours) - float(impact_hours))
                scenario_rows.append(
                    {
                        "scenario": "upper_bound_if_eliminated",
                        "value": modeled_value,
                        "delta": modeled_value - float(total_busy_hours),
                        "unit": "hours",
                    }
                )
        lines.append("Expected impact (current vs scenarios):")
        lines.append(
            f"Impact metric context: unit={unit}; population={denominator}; threshold/filters={filter_text}."
        )
        lines.append("| Scenario | Value | Delta | Unit |")
        lines.append("|---|---:|---:|---|")
        for row in scenario_rows[:10]:
            lines.append(
                "| {scenario} | {value:.3f} | {delta:.3f} | {unit} |".format(
                    scenario=row["scenario"],
                    value=float(row["value"]),
                    delta=float(row["delta"]),
                    unit=row.get("unit") or unit,
                )
            )
        lines.append(f"Confidence: {confidence_tag}")
        artifact_paths = _artifact_paths(report, rec.get("plugin_id"))
        artifact_text = ", ".join(artifact_paths) if artifact_paths else "n/a"
        filters = rec.get("query") or filter_text
        plugin_id = rec.get("plugin_id") or "n/a"
        lines.append(
            f"How to validate: re-run `{plugin_id}` and confirm `{metric_name}` in {artifact_text}; "
            f"filters: {filters}."
        )
        lines.append("")

    if known_items:
        lines.append("## Known-Issue Checks (Pass Gate)")
        if known_summary:
            lines.append(known_summary)
        known_items = _dedupe_recommendations_text(known_items)
        lines.append("| Status | Issue | Recommendation |")
        lines.append("|---|---|---|")
        for item in known_items[:10]:
            status = item.get("status") or "unknown"
            title = item.get("title") or "Known issue"
            rec_text = item.get("recommendation") or "Recommendation"
            lines.append(f"| {status} | {title} | {rec_text} |")
        lines.append("")

    atomic_write_text(run_dir / "business_summary.md", "\n".join(lines) + "\n")


def _write_engineering_summary(report: dict[str, Any], run_dir: Path) -> None:
    lines = ["# Engineering Summary", ""]
    lines.append("## Glossary")
    lines.append("- eligible_wait: time from eligible timestamp to start timestamp.")
    lines.append("- threshold: per-run wait threshold in seconds for defining over-threshold wait.")
    lines.append(
        "- busy period: hourly bucket (by queue timestamp) within the close-cycle window "
        "where wait-to-start exceeds the threshold."
    )
    lines.append("- close cycle window: day-of-month window used to tag close-cycle activity.")
    lines.append("")

    evaluations = _evaluate_known_issues(report)
    cards = _issue_cards(report)
    card_by_title = {card.get("title"): card for card in cards}
    known_items, discovery_items, _, _ = _split_recommendations(
        report.get("recommendations") if isinstance(report.get("recommendations"), dict) else None
    )
    discovery_items = _dedupe_recommendations_text(discovery_items)
    recs = sorted(discovery_items, key=_final_recommendation_sort_key)[:3]
    merged_notes: list[str] = []

    lines.append("## Traceability")
    if recs:
        lines.append("| Claim | Plugin | Kind | Measurement | Artifact Path | Query/Filter | Metric Context |")
        lines.append("|---|---|---|---|---|---|---|")
        for rec in recs:
            title = rec.get("title") or "Recommendation"
            merged_titles = [
                text
                for text in (rec.get("merged_titles") or [])
                if isinstance(text, str) and text
            ]
            claim_text = title
            if len(merged_titles) > 1:
                claim_text = f"{title} (+{len(merged_titles) - 1} merged)"
                merged_text = "; ".join(dict.fromkeys(merged_titles))
                merged_notes.append(f"{title}: {merged_text}")
            plugin_id = rec.get("plugin_id") or "n/a"
            kind = rec.get("kind") or "n/a"
            measurement = rec.get("measurement_type") or "n/a"
            artifact_paths = _artifact_paths(report, rec.get("plugin_id"))
            artifact_text = ", ".join(artifact_paths) if artifact_paths else "n/a"
            filters = rec.get("query") or rec.get("where") or rec.get("contains") or "n/a"
            metric_name = rec.get("metric_name") or "Metric"
            unit = rec.get("unit") or _metric_unit(metric_name, rec)
            denominator = rec.get("denominator") or "n/a"
            lines.append(
                f"| {claim_text} | `{plugin_id}` | {kind} | {measurement} | {artifact_text} | {filters} | "
                f"unit={unit}; population={denominator}; threshold/filters={filters} |"
            )
    else:
        lines.append("No traceability data available.")
    lines.append("")
    if merged_notes:
        lines.append("Merged claims:")
        for note in merged_notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("## Failing Checks")
    failing = [entry for entry in evaluations if entry.get("status") != "confirmed"]
    if not failing:
        lines.append("No failing checks.")
    else:
        for entry in failing[:10]:
            predicate = (
                f"plugin={entry.get('plugin_id')} kind={entry.get('kind')} "
                f"where={entry.get('where')} contains={entry.get('contains')} "
                f"min={entry.get('min_count')} max={entry.get('max_count')}"
            )
            matched = entry.get("matched") or []
            item = matched[0] if matched else {}
            computed = {
                "observed_count": entry.get("count"),
                "baseline": item.get("baseline_wait_hours")
                or item.get("eligible_wait_gt_hours_total")
                or item.get("baseline_median_sec"),
                "observed": item.get("modeled_wait_hours")
                or item.get("eligible_wait_gt_hours_modeled")
                or item.get("modeled_median_sec"),
            }
            card = card_by_title.get(entry.get("label")) or {}
            metric_name = card.get("metric_name") or entry.get("metric") or "Metric"
            unit = _metric_unit(metric_name, item if isinstance(item, dict) else None)
            denominator = card.get("denominator") or entry.get("denominator") or "n/a"
            filter_text = entry.get("where") or entry.get("contains") or "n/a"
            lines.append(f"- Predicate: {predicate}")
            lines.append(f"  - Computed: {computed}")
            lines.append(
                f"  - Metric context: unit={unit}; population={denominator}; "
                f"threshold/filters={filter_text}."
            )
            lines.append(f"  - Reason: {entry.get('status')}")
    atomic_write_text(run_dir / "engineering_summary.md", "\n".join(lines) + "\n")


def _write_appendix_raw(report: dict[str, Any], run_dir: Path) -> None:
    lines = ["# Appendix (Raw)", ""]
    for plugin_id in sorted(report.get("plugins", {}).keys()):
        data = report["plugins"][plugin_id]
        trimmed = dict(data)
        findings = data.get("findings") or []
        summary = _collapse_findings(findings, max_examples=10)
        trimmed["findings_summary"] = summary
        trimmed["findings_full_path"] = f"report.json -> plugins.{plugin_id}.findings"
        if summary.get("count", 0) > 10:
            trimmed["findings"] = []
        else:
            trimmed["findings"] = summary.get("top_examples") or []
        lines.append(f"## {plugin_id}")
        lines.append("```json")
        lines.append(json_dumps(trimmed))
        lines.append("```")
        lines.append("")
    atomic_write_text(run_dir / "appendix_raw.md", "\n".join(lines) + "\n")


def _write_slide_kit(report: dict[str, Any], run_dir: Path) -> None:
    kit_dir = run_dir / "slide_kit"
    kit_dir.mkdir(parents=True, exist_ok=True)
    info = _busy_periods_info(report, run_dir)
    busy_periods = info.get("busy_periods") or []
    threshold = info.get("threshold_seconds")
    threshold_val = int(threshold) if isinstance(threshold, (int, float)) else ""
    busy_periods_sorted = sorted(
        busy_periods,
        key=lambda b: float(b.get("wait_to_start_hours_total") or 0.0),
        reverse=True,
    )
    busy_periods_top = busy_periods_sorted[:10]
    busy_rows = []
    for row in busy_periods_top:
        busy_rows.append(
            {
                "period_start": row.get("period_start"),
                "period_end": row.get("period_end"),
                "wait_to_start_hours_total": float(row.get("wait_to_start_hours_total") or 0.0),
                "rows_total": int(row.get("rows_total") or 0),
                "rows_over_threshold": int(row.get("rows_over_threshold") or 0),
                "weekday": row.get("weekday"),
                "weekend": row.get("weekend"),
                "after_hours": row.get("after_hours"),
                "threshold_seconds": threshold_val,
            }
        )
    _write_csv(
        kit_dir / "busy_periods.csv",
        busy_rows,
        [
            "period_start",
            "period_end",
            "wait_to_start_hours_total",
            "rows_total",
            "rows_over_threshold",
            "weekday",
            "weekend",
            "after_hours",
            "threshold_seconds",
        ],
    )

    total_busy_hours = sum(
        float(item.get("wait_to_start_hours_total") or 0.0) for item in busy_periods_sorted
    )
    scenarios = [
        {
            "scenario": "current",
            "kpi_hours": total_busy_hours,
            "delta_hours": 0.0,
            "confidence": "Measured",
            "scale_factor": "",
            "unit": "hours",
            "population": "standalone_non_excluded",
            "threshold_seconds": threshold_val,
        }
    ]
    scale_factor = None
    plugin = report.get("plugins", {}).get("analysis_queue_delay_decomposition")
    if isinstance(plugin, dict):
        for item in plugin.get("findings") or []:
            if isinstance(item, dict) and item.get("kind") == "capacity_scale_model":
                scale_factor = item.get("scale_factor")
                break
    if isinstance(scale_factor, (int, float)) and total_busy_hours:
        modeled = float(total_busy_hours) * float(scale_factor)
        scenarios.append(
            {
                "scenario": "add_one_server",
                "kpi_hours": modeled,
                "delta_hours": modeled - float(total_busy_hours),
                "confidence": "Modeled",
                "scale_factor": float(scale_factor),
                "unit": "hours",
                "population": "standalone_non_excluded",
                "threshold_seconds": threshold_val,
            }
        )
    _write_csv(
        kit_dir / "scenario_summary.csv",
        scenarios,
        [
            "scenario",
            "kpi_hours",
            "delta_hours",
            "confidence",
            "scale_factor",
            "unit",
            "population",
            "threshold_seconds",
        ],
    )

    contributors = _top_process_contributors(report, limit=10)
    contrib_rows = []
    for row in contributors:
        contrib_rows.append(
            {
                "process": row.get("process"),
                "eligible_wait_gt_hours_total": row.get("eligible_wait_gt_hours_total"),
                "runs_total": row.get("runs_total"),
                "share_of_total": row.get("share_of_total"),
                "unit": "hours",
                "population": "standalone_non_excluded",
                "threshold_seconds": threshold_val,
            }
        )
    _write_csv(
        kit_dir / "top_process_contributors.csv",
        contrib_rows,
        [
            "process",
            "eligible_wait_gt_hours_total",
            "runs_total",
            "share_of_total",
            "unit",
            "population",
            "threshold_seconds",
        ],
    )

    qemail = _qemail_profile(report)
    qemail_rows = []
    if qemail:
        qemail_rows.append(
            {
                "process": qemail.get("process"),
                "eligible_wait_gt_hours_total": qemail.get("eligible_wait_gt_hours_total"),
                "eligible_wait_hours_total": qemail.get("eligible_wait_hours_total"),
                "runs_total": qemail.get("runs_total"),
                "runs_close": qemail.get("runs_close"),
                "runs_open": qemail.get("runs_open"),
                "unit": "hours",
                "population": "standalone_non_excluded",
                "threshold_seconds": threshold_val,
            }
        )
    _write_csv(
        kit_dir / "qemail_profile.csv",
        qemail_rows,
        [
            "process",
            "eligible_wait_gt_hours_total",
            "eligible_wait_hours_total",
            "runs_total",
            "runs_close",
            "runs_open",
            "unit",
            "population",
            "threshold_seconds",
        ],
    )
