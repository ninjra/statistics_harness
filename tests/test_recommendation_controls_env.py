from __future__ import annotations

from statistic_harness.core.report import _build_discovery_recommendations


def _base_report() -> dict:
    return {
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
                        "evidence": {"process_norm": "rpt_por002"},
                        "measurement_type": "measured",
                    },
                    {
                        "kind": "actionable_ops_lever",
                        "process_norm": "qemail",
                        "title": "Reschedule qemail",
                        "recommendation": "Reschedule qemail to a lower-load hour.",
                        "action_type": "reschedule",
                        "expected_delta_seconds": 10800.0,
                        "confidence": 0.9,
                        "evidence": {"process_norm": "qemail"},
                        "measurement_type": "measured",
                    },
                ]
            }
        }
    }


def test_allow_action_types_and_top_n(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_ALLOW_ACTION_TYPES", "batch_input")
    monkeypatch.setenv("STAT_HARNESS_DISCOVERY_TOP_N", "1")
    report = _base_report()
    discovery = _build_discovery_recommendations(report, storage=None, run_dir=None)
    items = discovery.get("items") if isinstance(discovery, dict) else None
    assert isinstance(items, list)
    assert len(items) == 1
    assert items[0].get("action_type") == "batch_input"
    assert items[0].get("where", {}).get("process_norm") == "rpt_por002"


def test_allow_process_patterns_filters_discovery(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_RECOMMENDATION_ALLOW_PROCESSES", "rpt_*")
    report = _base_report()
    discovery = _build_discovery_recommendations(report, storage=None, run_dir=None)
    items = discovery.get("items") if isinstance(discovery, dict) else None
    assert isinstance(items, list)
    assert len(items) == 1
    assert items[0].get("where", {}).get("process_norm") == "rpt_por002"


def test_default_obviousness_filter_hides_obvious_actions(monkeypatch) -> None:
    monkeypatch.delenv("STAT_HARNESS_ALLOW_ACTION_TYPES", raising=False)
    monkeypatch.delenv("STAT_HARNESS_MAX_OBVIOUSNESS", raising=False)
    report = {
        "plugins": {
            "analysis_actionable_ops_levers_v1": {
                "findings": [
                    {
                        "kind": "actionable_ops_lever",
                        "process_norm": "rpt_por002",
                        "title": "Batch payout input",
                        "recommendation": "Convert process_id to batch input.",
                        "action_type": "batch_group_candidate",
                        "expected_delta_seconds": 300.0,
                        "confidence": 0.9,
                        "evidence": {"process_norm": "rpt_por002"},
                        "measurement_type": "measured",
                    },
                    {
                        "kind": "actionable_ops_lever",
                        "process_norm": "close_cycle",
                        "title": "Reduce spillover",
                        "recommendation": "Reduce spillover past EOM.",
                        "action_type": "reduce_spillover_past_eom",
                        "expected_delta_seconds": 1000.0,
                        "confidence": 0.9,
                        "evidence": {"process_norm": "close_cycle"},
                        "measurement_type": "measured",
                    },
                ]
            }
        }
    }
    discovery = _build_discovery_recommendations(report, storage=None, run_dir=None)
    items = discovery.get("items") if isinstance(discovery, dict) else None
    assert isinstance(items, list)
    assert len(items) == 1
    assert items[0].get("action_type") == "batch_group_candidate"
    assert items[0].get("obviousness_rank") == "needle"


def test_ideaspace_actions_are_surfaced_with_action_types(monkeypatch) -> None:
    monkeypatch.delenv("STAT_HARNESS_ALLOW_ACTION_TYPES", raising=False)
    monkeypatch.delenv("STAT_HARNESS_MAX_OBVIOUSNESS", raising=False)
    report = {
        "plugins": {
            "analysis_ideaspace_action_planner": {
                "findings": [
                    {
                        "kind": "ideaspace_action",
                        "title": "Add QPEC capacity (+1) to reduce queue delays",
                        "what": "Action: QPEC hosts show high eligible-wait pressure; add one QPEC server (QPEC+1).",
                        "lever_id": "add_qpec_capacity_plus_one_v1",
                        "action_type": "add_server",
                        "target": "qpec",
                        "delta_value": 33.3,
                        "confidence": 0.7,
                        "measurement_type": "modeled",
                        "evidence": {"metrics": {"qpec_host_count": 2}},
                    },
                    {
                        "kind": "ideaspace_action",
                        "title": "Tune QEMAIL schedule frequency",
                        "what": "Action: Increase QEMAIL schedule interval from 5 to 15 minutes.",
                        "lever_id": "tune_schedule_qemail_frequency_v1",
                        "action_type": "tune_schedule",
                        "target": "qemail",
                        "delta_value": 12.5,
                        "confidence": 0.65,
                        "measurement_type": "modeled",
                        "evidence": {"metrics": {"median_interval_sec": 300.0}},
                    },
                ]
            }
        }
    }
    discovery = _build_discovery_recommendations(report, storage=None, run_dir=None)
    items = discovery.get("items") if isinstance(discovery, dict) else None
    assert isinstance(items, list)
    assert any(i.get("action_type") == "add_server" for i in items)
    assert any(i.get("action_type") == "tune_schedule" for i in items)
