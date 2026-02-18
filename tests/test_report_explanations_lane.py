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
