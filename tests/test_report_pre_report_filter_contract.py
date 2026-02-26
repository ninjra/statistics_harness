from __future__ import annotations

from statistic_harness.core.report import _build_recommendations


def test_pre_report_passthrough_keeps_all_candidates(monkeypatch) -> None:
    monkeypatch.delenv("STAT_HARNESS_PRE_REPORT_FILTER_MODE", raising=False)
    monkeypatch.delenv("STAT_HARNESS_REQUIRE_MODELED_HOURS", raising=False)
    monkeypatch.delenv("STAT_HARNESS_REQUIRE_DIRECT_PROCESS_ACTION", raising=False)
    report = {
        "plugins": {
            "analysis_actionable_ops_levers_v1": {
                "findings": [
                    {
                        "kind": "actionable_ops_lever",
                        "process_norm": "proc_keep_a",
                        "title": "A",
                        "recommendation": "A",
                        "action_type": "batch_input",
                        "expected_delta_seconds": 3600.0,
                        "confidence": 0.9,
                        "measurement_type": "modeled",
                    },
                    {
                        "kind": "actionable_ops_lever",
                        "process_norm": "proc_keep_b",
                        "title": "B",
                        "recommendation": "B",
                        "action_type": "review",
                        "expected_delta_seconds": 0.0,
                        "confidence": 0.8,
                        "measurement_type": "modeled",
                    },
                ]
            }
        }
    }

    payload = _build_recommendations(report)
    discovery = payload.get("discovery") or {}
    items = discovery.get("items") or []

    assert discovery.get("pre_report_filter_mode") == "passthrough"
    assert discovery.get("dropped_non_direct_process") == 0
    assert discovery.get("dropped_without_modeled_hours") == 0
    assert discovery.get("dropped_by_action_cap") == 0
    assert discovery.get("top_n_applied") is False
    assert int(discovery.get("candidate_count_before_top_n") or 0) == len(items)
    processes = {
        str((row.get("where") or {}).get("process_norm") or "").strip().lower()
        for row in items
        if isinstance(row, dict)
    }
    assert "proc_keep_a" in processes
    assert "proc_keep_b" in processes


def test_pre_report_strict_can_drop_candidates(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_PRE_REPORT_FILTER_MODE", "strict")
    monkeypatch.setenv("STAT_HARNESS_REQUIRE_MODELED_HOURS", "1")
    monkeypatch.setenv("STAT_HARNESS_REQUIRE_DIRECT_PROCESS_ACTION", "1")
    report = {
        "plugins": {
            "analysis_actionable_ops_levers_v1": {
                "findings": [
                    {
                        "kind": "actionable_ops_lever",
                        "process_norm": "proc_direct",
                        "title": "Direct",
                        "recommendation": "Direct",
                        "action_type": "batch_input",
                        "expected_delta_seconds": 3600.0,
                        "confidence": 0.9,
                        "measurement_type": "modeled",
                    },
                    {
                        "kind": "actionable_ops_lever",
                        "process_norm": "proc_drop",
                        "title": "Drop",
                        "recommendation": "Drop",
                        "action_type": "review",
                        "expected_delta_seconds": 0.0,
                        "confidence": 0.8,
                        "measurement_type": "modeled",
                    },
                ]
            }
        }
    }

    payload = _build_recommendations(report)
    discovery = payload.get("discovery") or {}
    items = discovery.get("items") or []
    before = int(discovery.get("candidate_count_before_top_n") or 0)

    assert discovery.get("pre_report_filter_mode") == "strict"
    assert before >= len(items)
    assert int(discovery.get("dropped_non_direct_process") or 0) >= 0
    assert int(discovery.get("dropped_without_modeled_hours") or 0) >= 0
