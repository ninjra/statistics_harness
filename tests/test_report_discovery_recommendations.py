import os

import statistic_harness.core.report as report_module
from statistic_harness.core.report import _build_recommendations


def test_discovery_kept_when_known_issues_have_no_expected_findings(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_REQUIRE_DIRECT_PROCESS_ACTION", "0")
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


def test_discovery_filter_exclusions_and_instrumentation_limit(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_REQUIRE_DIRECT_PROCESS_ACTION", "0")
    os.environ["STAT_HARNESS_EXCLUDED_PROCESSES"] = "*los*,qpec"
    try:
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
                            "estimated_delta_hours_total": 1.0,
                        },
                        {
                            "kind": "ideaspace_action",
                            "process_id": "proc_ok2",
                            "recommendation": "add trace instrumentation second",
                            "estimated_delta_hours_total": 0.5,
                        },
                    ]
                }
            }
        }
        payload = _build_recommendations(report)
        recs = payload["items"]
        assert len(recs) == 1
        assert recs[0]["process_id"] == "proc_ok"
    finally:
        del os.environ["STAT_HARNESS_EXCLUDED_PROCESSES"]


def test_discovery_recommendations_require_positive_modeled_hours() -> None:
    report = {
        "plugins": {
            "analysis_queue_delay_decomposition": {
                "findings": [
                    {
                        "kind": "eligible_wait_impact",
                        "eligible_wait_hours_total": 52.0,
                    }
                ]
            },
            "analysis_actionable_ops_levers_v1": {
                "findings": [
                    {
                        "kind": "actionable_ops_lever",
                        "process_norm": "rpt_por002",
                        "title": "Batch payout input",
                        "recommendation": "Convert process_id `rpt_por002` to batch input.",
                        "action_type": "batch_input",
                        "expected_delta_seconds": 7200.0,
                        "confidence": 0.9,
                        "measurement_type": "modeled",
                    },
                    {
                        "kind": "actionable_ops_lever",
                        "process_norm": "pognrtrpt",
                        "title": "Zero delta should be removed",
                        "recommendation": "This should not survive modeled-hours gating.",
                        "action_type": "batch_input",
                        "expected_delta_seconds": 0.0,
                        "confidence": 0.9,
                        "measurement_type": "modeled",
                    },
                ]
            },
            "analysis_ideaspace_action_planner": {
                "findings": [
                    {
                        "kind": "ideaspace_action",
                        "title": "Add QPEC capacity (+1) to reduce queue delays",
                        "what": "Action: QPEC hosts show high eligible-wait pressure; add one QPEC server.",
                        "lever_id": "add_qpec_capacity_plus_one_v1",
                        "action_type": "add_server",
                        "target": "qpec",
                        "delta_value": 33.3333333333,
                        "unit": "percent",
                        "confidence": 0.7,
                        "measurement_type": "modeled",
                        "evidence": {"metrics": {"qpec_host_count": 2}},
                    }
                ]
            },
        }
    }

    payload = _build_recommendations(report)
    items = payload["items"]
    assert items
    assert all(float(i.get("modeled_delta_hours") or 0.0) > 0.0 for i in items)
    assert all(i.get("optimization_metric") == "modeled_user_hours_saved" for i in items)
    assert all(i.get("not_modeled_reason") is None for i in items)
    assert all("pognrtrpt" not in str(i.get("where", {}).get("process_norm", "")) for i in items)


def test_dependency_chain_flow_actions_are_not_actionable() -> None:
    report = {
        "plugins": {
            "analysis_actionable_ops_levers_v1": {
                "findings": [
                    {
                        "kind": "actionable_ops_lever",
                        "process_norm": "jboachild",
                        "title": "Unblock jboachild when preceded by jbownalloc",
                        "recommendation": "legacy child-targeted recommendation",
                        "action_type": "unblock_dependency_chain",
                        "expected_delta_seconds": 3600.0,
                        "confidence": 0.8,
                        "measurement_type": "modeled",
                        "evidence": {
                            "parent_process": "jbownalloc",
                            "child_process": "jboachild",
                            "child_dependency_non_null_ratio": 1.0,
                        },
                    }
                ]
            }
        }
    }
    payload = _build_recommendations(report)
    items = payload["items"]
    assert not items


def test_chain_bound_processes_are_filtered_from_direct_actions(monkeypatch) -> None:
    report = {
        "plugins": {
            "analysis_actionable_ops_levers_v1": {
                "findings": [
                    {
                        "kind": "actionable_ops_lever",
                        "process_norm": "rpt_por002",
                        "title": "Batch payout input",
                        "recommendation": "Convert process_id `rpt_por002` to batch input.",
                        "action_type": "batch_input",
                        "expected_delta_seconds": 7200.0,
                        "confidence": 0.9,
                        "measurement_type": "modeled",
                    },
                    {
                        "kind": "actionable_ops_lever",
                        "process_norm": "jbjeminhld",
                        "title": "Throttle burst arrivals",
                        "recommendation": "Throttle/de-duplicate burst arrivals for jbjeminhld.",
                        "action_type": "throttle_or_dedupe",
                        "expected_delta_seconds": 14400.0,
                        "confidence": 0.9,
                        "measurement_type": "modeled",
                    },
                ]
            }
        }
    }
    monkeypatch.setattr(
        report_module,
        "_process_chain_modifiability_map",
        lambda _report, _storage, _process_hints: {
            "rpt_por002": {"chain_bound": False, "chain_bound_ratio": 0.0},
            "jbjeminhld": {"chain_bound": True, "chain_bound_ratio": 1.0},
        },
    )
    payload = _build_recommendations(report)
    items = payload["items"]
    assert len(items) == 1
    assert items[0].get("where", {}).get("process_norm") == "rpt_por002"
    assert "chain-bound" in str(payload.get("discovery", {}).get("summary") or "")
