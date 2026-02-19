from __future__ import annotations

from statistic_harness.core.report import _build_recommendations


def test_recommendations_include_non_actionable_explanations_with_downstream_lists(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_REQUIRE_DIRECT_PROCESS_ACTION", "0")
    monkeypatch.setenv("STAT_HARNESS_REQUIRE_MODELED_HOURS", "0")
    report = {
        "plugins": {
            "analysis_ideaspace_action_planner": {
                "status": "ok",
                "summary": "Planner produced one candidate",
                "findings": [
                    {
                        "kind": "ideaspace_action",
                        "process_id": "proc_x",
                        "recommendation": "Add capacity for proc_x.",
                    }
                ],
            },
            "profile_basic": {
                "status": "ok",
                "summary": "Profiled baseline columns",
                "findings": [{"kind": "profile_overview", "measurement_type": "measured"}],
            },
            "analysis_association_rules_apriori_v1": {
                "status": "ok",
                "summary": "No strong rules crossed confidence threshold",
                "findings": [],
            },
        }
    }

    payload = _build_recommendations(report)
    explanations = payload.get("explanations")
    assert isinstance(explanations, dict)
    items = explanations.get("items")
    assert isinstance(items, list) and items

    by_plugin = {
        str(item.get("plugin_id")): item for item in items if isinstance(item, dict)
    }
    # Actionable plugin should not be forced into explanation lane.
    assert "analysis_ideaspace_action_planner" not in by_plugin

    profile_item = by_plugin.get("profile_basic")
    assert isinstance(profile_item, dict)
    assert str(profile_item.get("plugin_type")) == "profile"
    assert isinstance(profile_item.get("plain_english_explanation"), str)
    assert profile_item.get("plain_english_explanation")
    assert isinstance(profile_item.get("recommended_next_step"), str)
    assert isinstance(profile_item.get("downstream_plugins"), list)
    assert "analysis_association_rules_apriori_v1" in profile_item.get("downstream_plugins")

    coverage = payload.get("actionability_coverage")
    assert isinstance(coverage, dict)
    assert int(coverage.get("unexplained_plugin_count") or 0) == 0


def test_non_actionable_reason_is_no_decision_signal_for_non_recommendation_findings(
    monkeypatch,
) -> None:
    monkeypatch.setenv("STAT_HARNESS_REQUIRE_DIRECT_PROCESS_ACTION", "1")
    monkeypatch.setenv("STAT_HARNESS_REQUIRE_MODELED_HOURS", "1")
    report = {
        "plugins": {
            "analysis_cluster_analysis_auto": {
                "status": "ok",
                "summary": "Cluster profile produced diagnostics only",
                "findings": [{"kind": "cluster_analysis_auto"}],
            }
        }
    }
    payload = _build_recommendations(report)
    explanations = payload.get("explanations") if isinstance(payload, dict) else {}
    items = explanations.get("items") if isinstance(explanations, dict) else []
    row = next(
        (
            item
            for item in (items or [])
            if isinstance(item, dict) and item.get("plugin_id") == "analysis_cluster_analysis_auto"
        ),
        None,
    )
    assert isinstance(row, dict)
    assert row.get("reason_code") == "NO_ACTIONABLE_FINDING_CLASS"


def test_non_actionable_reason_flags_missing_direct_process_target(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_REQUIRE_DIRECT_PROCESS_ACTION", "1")
    monkeypatch.setenv("STAT_HARNESS_REQUIRE_MODELED_HOURS", "0")
    report = {
        "plugins": {
            "analysis_actionable_ops_levers_v1": {
                "status": "ok",
                "summary": "Action combo plan",
                "findings": [
                    {
                        "kind": "actionable_ops_lever",
                        "process_norm": "(multiple)",
                        "recommendation": "Execute the selected actions as one package.",
                        "action_type": "action_plan_combo",
                        "expected_delta_seconds": 3600.0,
                        "measurement_type": "modeled",
                    }
                ],
            }
        }
    }
    payload = _build_recommendations(report)
    explanations = payload.get("explanations") if isinstance(payload, dict) else {}
    items = explanations.get("items") if isinstance(explanations, dict) else []
    row = next(
        (
            item
            for item in (items or [])
            if isinstance(item, dict)
            and item.get("plugin_id") == "analysis_actionable_ops_levers_v1"
        ),
        None,
    )
    assert isinstance(row, dict)
    assert row.get("reason_code") == "NO_DIRECT_PROCESS_TARGET"


def test_non_actionable_reason_flags_plugin_precondition_unmet(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_REQUIRE_DIRECT_PROCESS_ACTION", "1")
    monkeypatch.setenv("STAT_HARNESS_REQUIRE_MODELED_HOURS", "0")
    report = {
        "plugins": {
            "analysis_ideaspace_action_planner": {
                "status": "ok",
                "summary": "deterministic fallback",
                "findings": [
                    {
                        "kind": "plugin_not_applicable",
                        "what": "plugin failed",
                        "why": "preconditions not satisfied",
                        "required_inputs": ["process_norm", "duration_seconds"],
                        "missing_inputs": ["duration_seconds"],
                    }
                ],
            }
        }
    }
    payload = _build_recommendations(report)
    explanations = payload.get("explanations") if isinstance(payload, dict) else {}
    items = explanations.get("items") if isinstance(explanations, dict) else []
    row = next(
        (
            item
            for item in (items or [])
            if isinstance(item, dict)
            and item.get("plugin_id") == "analysis_ideaspace_action_planner"
        ),
        None,
    )
    assert isinstance(row, dict)
    assert row.get("reason_code") == "PLUGIN_PRECONDITION_UNMET"
    assert row.get("missing_inputs") == ["duration_seconds"]
    assert "duration_seconds" in str(row.get("recommended_next_step") or "")


def test_non_actionable_reason_flags_observation_only(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_REQUIRE_DIRECT_PROCESS_ACTION", "1")
    monkeypatch.setenv("STAT_HARNESS_REQUIRE_MODELED_HOURS", "0")
    report = {
        "plugins": {
            "analysis_percentile_analysis": {
                "status": "ok",
                "summary": "observation only",
                "findings": [{"kind": "plugin_observation", "what": "diagnostic"}],
            }
        }
    }
    payload = _build_recommendations(report)
    explanations = payload.get("explanations") if isinstance(payload, dict) else {}
    items = explanations.get("items") if isinstance(explanations, dict) else []
    row = next(
        (
            item
            for item in (items or [])
            if isinstance(item, dict) and item.get("plugin_id") == "analysis_percentile_analysis"
        ),
        None,
    )
    assert isinstance(row, dict)
    assert row.get("reason_code") == "OBSERVATION_ONLY"
