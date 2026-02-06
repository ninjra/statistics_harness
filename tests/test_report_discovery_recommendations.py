import os

from statistic_harness.core.report import _build_recommendations


def test_discovery_kept_when_known_issues_have_no_expected_findings() -> None:
    report = {
        "known_issues": {"expected_findings": []},
        "plugins": {
            "analysis_ideaspace_action_planner": {
                "findings": [
                    {
                        "kind": "ideaspace_action",
                        "process_id": "proc_x",
                        "recommendation": "Target process proc_x: isolate capacity.",
                        "estimated_delta_hours_total": 2.0,
                        "estimated_delta_pct": 0.2,
                        "estimated_delta_seconds": 120,
                        "evidence": {"runs": 20, "gap_sec": 480},
                        "validation_steps": ["a", "b"],
                        "action_type": "operational_procedure",
                    }
                ]
            }
        },
    }
    payload = _build_recommendations(report)
    assert payload["status"] == "ok"
    assert payload["items"]
    assert payload["items"][0]["status"] == "discovery"


def test_discovery_filter_exclusions_and_instrumentation_limit() -> None:
    os.environ["STAT_HARNESS_EXCLUDED_PROCESSES"] = "*los*,qpec"
    report = {
        "plugins": {
            "analysis_ideaspace_action_planner": {
                "findings": [
                    {
                        "kind": "ideaspace_action",
                        "process_id": "qLOS_one",
                        "recommendation": "excluded",
                    },
                    {
                        "kind": "ideaspace_action",
                        "process_id": "proc_ok",
                        "recommendation": "add instrumentation for queue",
                    },
                    {
                        "kind": "ideaspace_action",
                        "process_id": "proc_ok2",
                        "recommendation": "add trace instrumentation second",
                    },
                ]
            }
        }
    }
    payload = _build_recommendations(report)
    recs = payload["items"]
    assert len(recs) == 1
    assert recs[0]["process_id"] == "proc_ok"
    del os.environ["STAT_HARNESS_EXCLUDED_PROCESSES"]
