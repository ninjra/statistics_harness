from __future__ import annotations

from statistic_harness.core.report import _discovery_recommendation_sort_key


def test_structural_action_types_sort_before_operational_tuning() -> None:
    batch = {
        "plugin_id": "analysis_actionable_ops_levers_v1",
        "kind": "actionable_ops_lever",
        "action_type": "batch_input",
        "relevance_score": 1.0,
        "impact_hours": 1.0,
    }
    schedule = {
        "plugin_id": "analysis_actionable_ops_levers_v1",
        "kind": "actionable_ops_lever",
        "action_type": "reschedule",
        "relevance_score": 1.0,
        "impact_hours": 1.0,
    }
    assert _discovery_recommendation_sort_key(batch) > _discovery_recommendation_sort_key(schedule)

