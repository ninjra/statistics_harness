"""Reporting package — decomposed from the monolithic report.py.

Each submodule owns a cohesive slice of the report-building pipeline.
This ``__init__`` re-exports every public and private name so that
existing ``from statistic_harness.core.report import X`` paths keep
working via the thin facade in ``report.py``.
"""
from __future__ import annotations

# -- config: env-var readers, feature flags, thresholds ----------------
from .config import (  # noqa: F401
    _ALLOW_ACTION_TYPES_ENV,
    _ALLOW_KNOWN_ISSUE_SYNTHETIC_MATCHES_ENV,
    _ALLOW_PROCESS_PATTERNS_ENV,
    _CHAIN_BOUND_RATIO_MIN_ENV,
    _CLOSE_CYCLES_PER_YEAR_ENV,
    _DIRECT_PROCESS_ACTION_TYPES_ENV,
    _DISCOVERY_TOP_N_ENV,
    _INCLUDE_KNOWN_RECOMMENDATIONS_ENV,
    _MAX_OBVIOUSNESS_ENV,
    _MAX_PER_ACTION_TYPE_ENV,
    _MIN_RELEVANCE_SCORE_ENV,
    _PLUGIN_CLASS_TAXONOMY_CACHE,
    _PLUGIN_CLASS_TAXONOMY_PATH,
    _PRE_REPORT_FILTER_MODE_ENV,
    _RANKING_VERSION_ENV,
    _RANKING_WEIGHTS_CACHE,
    _RANKING_WEIGHTS_PATH,
    _RECENCY_DECAY_PER_MONTH_ENV,
    _RECENCY_MIN_WEIGHT_ENV,
    _RECENCY_UNKNOWN_WEIGHT_ENV,
    _REPORTING_CONFIG_CACHE,
    _REPORTING_CONFIG_PATH,
    _REPORT_MD_MAX_EVIDENCE_IDS_ENV,
    _REPORT_MD_MAX_FINDINGS_PER_PLUGIN_ENV,
    _REPORT_MD_MAX_STRING_LEN_ENV,
    _REQUIRE_DIRECT_PROCESS_ACTION_ENV,
    _REQUIRE_MODELED_HOURS_ENV,
    _SUPPRESS_ACTION_TYPES_ENV,
    _allow_action_types,
    _allow_known_issue_synthetic_matches,
    _allow_process_patterns,
    _chain_bound_ratio_min,
    _close_cycles_per_year,
    _direct_process_action_types,
    _discovery_top_n,
    _env_float,
    _include_capacity_recommendations,
    _include_known_recommendations,
    _max_obviousness,
    _max_per_action_type,
    _min_relevance_score,
    _pre_report_filter_mode,
    _pre_report_passthrough_enabled,
    _ranking_version,
    _read_positive_int_env,
    _recommendation_controls,
    _reporting_config,
    _require_direct_process_action,
    _require_modeled_hours,
    _suppressed_action_types,
)

# -- action_types: classification, normalisation, tiering -------------
from .action_types import (  # noqa: F401
    _action_type_obviousness,
    _action_type_tier,
    _flow_rewire_action_types,
    _infer_action_type_from_text,
    _normalize_action_type,
    _obviousness_rank,
    _tier_score_for_item,
)

# -- process_targeting: process hint extraction and targeting ----------
from .process_targeting import (  # noqa: F401
    _extract_process_queue_ids_from_finding,
    _finding_recommendation_text,
    _has_backstop_decision_signal,
    _infer_process_from_text,
    _known_process_terms_from_report,
    _normalize_process_hint,
    _process_hint,
    _recommendation_process_hint,
    _target_process_ids_for_finding,
)

# -- matching: pattern/value matching, alias resolution ----------------
from .matching import (  # noqa: F401
    _collect_alias_values,
    _match_key_aliases,
    _matches_contains_value,
    _matches_expected,
    _matches_expected_impl,
    _matches_where_value,
    _normalize_match_value,
)

# -- manifest: plugin metadata loading --------------------------------
from .manifest import (  # noqa: F401
    _downstream_consumers,
    _extract_precondition_inputs,
    _manifest_index,
    _plugin_class_id,
    _plugin_class_taxonomy,
    _plugin_expected_output_type,
)

# -- ideaspace: column role inference ----------------------------------
from .ideaspace import (  # noqa: F401
    _ideaspace_families_summary,
    _infer_ideaspace_roles,
)

# -- known_issues: known issue matching, synthetic findings ------------
from .known_issues import (  # noqa: F401
    _collect_findings_for_plugin,
    _evaluate_known_issues,
    _known_issue_processes,
    _known_recommendation_match,
    _load_known_issues_fallback,
    _sanitize_known_recommendation_exclusions,
)

# -- text: recommendation text, trimming, formatting -------------------
from .text import (  # noqa: F401
    _collapse_findings,
    _format_findings,
    _format_issue_value,
    _format_known_issue_checks,
    _format_metrics,
    _format_plugin_table,
    _format_value,
    _plugin_summary_rows,
    _recommendation_text,
    _trim_finding_for_markdown,
    _trim_long_text,
    _trim_plugin_dump_for_markdown,
    _write_csv,
)

# -- scoring: sort keys, dedup, confidence/controllability weights -----
from .scoring import (  # noqa: F401
    _capacity_scale_recommendation,
    _confidence_weight,
    _controllability_weight,
    _dedupe_recommendations,
    _dedupe_recommendations_text,
    _discovery_recommendation_sort_key,
    _final_recommendation_sort_key,
    _item_process_norm,
    _modeled_improvement_percent,
    _recommendation_merge_key,
    _safe_num,
    _split_recommendations,
)

# -- artifacts: evidence extraction, analytics, slide kit --------------
from .artifacts import (  # noqa: F401
    _artifact_paths,
    _build_executive_summary,
    _denominator_text,
    _impact_hours,
    _issue_cards,
    _item_evidence_row,
    _load_artifact_json,
    _metric_spec,
    _metric_unit,
    _primary_process_for_item,
    _queue_delay_results,
    _scope_class_for_item,
    _scope_type_for_item,
    _target_process_ids_for_item,
    _waterfall_summary,
)

# -- traceability: claim registry for rendered numbers ------------------
from .traceability import Claim, ClaimRegistry  # noqa: F401

# -- redaction: scrub hostnames, user IDs, sensitive data ---------------
from .redaction import (  # noqa: F401
    FORBIDDEN_COLUMNS,
    check_forbidden_columns,
    pseudonymize,
    redact_dict_values,
    redact_hostnames,
    redact_user_ids,
)

# -- guardrails: post-build fail-fast checks ----------------------------
from .guardrails import (  # noqa: F401
    GuardrailViolation,
    check_forbidden_slide_kit_columns,
    check_recommendation_dedupe_conflicts,
    check_unclaimed_numbers,
    check_waterfall_reconciliation,
    run_all_guardrails,
)
