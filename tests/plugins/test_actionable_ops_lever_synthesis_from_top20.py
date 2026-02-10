from __future__ import annotations


def test_report_synthesizes_actionable_ops_levers_from_any_plugin() -> None:
    # Ensure the report layer does not hardcode a single plugin_id for actionable levers.
    from statistic_harness.core.report import _build_discovery_recommendations

    report = {
        "config": {"exclude_processes": []},
        "plugins": {
            "analysis_param_near_duplicate_minhash_v1": {
                "findings": [
                    {
                        "kind": "actionable_ops_lever",
                        "process_norm": "rpt_por002",
                        "process_id": "proc:rpt_por002",
                        "title": "Batch input candidate",
                        "recommendation": "Add a batch input mode for payout_id.",
                        "action_type": "batch_input_refactor",
                        "expected_delta_seconds": 3600.0,
                        "confidence": 0.8,
                        "evidence": {"process_norm": "rpt_por002"},
                        "measurement_type": "measured",
                    }
                ]
            }
        },
    }
    discovery = _build_discovery_recommendations(report, storage=None, run_dir=None)
    items = discovery.get("items") if isinstance(discovery, dict) else None
    assert isinstance(items, list)
    assert any(
        isinstance(it, dict)
        and it.get("kind") == "actionable_ops_lever"
        and it.get("plugin_id") == "analysis_param_near_duplicate_minhash_v1"
        for it in items
    )

