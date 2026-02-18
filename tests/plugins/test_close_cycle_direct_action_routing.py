from __future__ import annotations

from statistic_harness.core.report import _build_discovery_recommendations


def test_close_cycle_uplift_findings_are_routed_to_discovery_actions(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_REQUIRE_DIRECT_PROCESS_ACTION", "0")
    report = {
        "plugins": {
            "analysis_close_cycle_uplift": {
                "findings": [
                    {
                        "kind": "close_cycle_share_shift",
                        "process": "QEMAIL",
                        "process_norm": "qemail",
                        "close_share": 0.28,
                        "open_share": 0.20,
                        "share_delta": 0.08,
                        "close_count": 120,
                        "median_close": 2400.0,
                        "median_open": 1200.0,
                        "slowdown_ratio": 2.0,
                        "p_value": 0.01,
                    }
                ]
            }
        }
    }
    discovery = _build_discovery_recommendations(report, storage=None, run_dir=None)
    items = discovery.get("items") if isinstance(discovery, dict) else None
    assert isinstance(items, list)
    routed = [
        item
        for item in items
        if item.get("plugin_id") == "analysis_close_cycle_uplift"
        and item.get("kind") == "close_cycle_share_shift"
    ]
    assert routed
    assert routed[0].get("action_type") == "reduce_transition_gap"
    assert float(routed[0].get("modeled_percent_hint") or 0.0) > 0.0


def test_close_cycle_capacity_model_routes_only_when_modeled_improves(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_REQUIRE_DIRECT_PROCESS_ACTION", "0")
    report = {
        "plugins": {
            "analysis_close_cycle_capacity_model": {
                "findings": [
                    {
                        "kind": "close_cycle_capacity_model",
                        "decision": "modeled",
                        "host_metric": "concurrent",
                        "metric_type": "queue_to_end",
                        "baseline_value": 3600.0,
                        "modeled_value": 1800.0,
                        "measurement_type": "modeled",
                    },
                    {
                        "kind": "close_cycle_capacity_model",
                        "decision": "modeled",
                        "host_metric": "unique",
                        "metric_type": "queue_to_end",
                        "baseline_value": 1800.0,
                        "modeled_value": 1800.0,
                        "measurement_type": "modeled",
                    },
                ]
            }
        }
    }
    discovery = _build_discovery_recommendations(report, storage=None, run_dir=None)
    items = discovery.get("items") if isinstance(discovery, dict) else None
    assert isinstance(items, list)
    routed = [
        item
        for item in items
        if item.get("plugin_id") == "analysis_close_cycle_capacity_model"
        and item.get("kind") == "close_cycle_capacity_model"
    ]
    assert len(routed) == 1
    assert routed[0].get("action_type") == "add_server"
    assert float(routed[0].get("modeled_percent_hint") or 0.0) > 0.0
