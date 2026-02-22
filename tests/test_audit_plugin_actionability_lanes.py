from __future__ import annotations

from scripts.audit_plugin_actionability import _classify_next_step_lane


def test_classify_next_step_lane_covers_standard_phrases() -> None:
    cases = {
        "Fix the plugin failure and rerun the full gauntlet.": "failure_recovery_contract",
        "Verify input prerequisites; if this should apply, update plugin gating and rerun.": "prerequisite_contract",
        "Normalize finding kind values for this plugin so routing can map outputs to actions.": "finding_schema_contract",
        "Confirm data coverage and add plugin-native findings or explanation mapping so this plugin contributes decision support.": "data_coverage_contract",
    }
    for next_step, expected_lane in cases.items():
        lane, post_state = _classify_next_step_lane(
            {
                "actionability_state": "explained_non_actionable",
                "recommended_next_step": next_step,
            }
        )
        assert lane == expected_lane
        assert post_state == "actionable_or_deterministic_explained_non_actionable"

