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


def test_discovery_skips_qemail_model_when_qemail_is_excluded(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_EXCLUDED_PROCESSES", "qemail")
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
                    }
                ]
            }
        },
    }
    calls = {"count": 0}

    def _raise_if_called(*_args, **_kwargs):
        calls["count"] += 1
        raise AssertionError("qemail model should be skipped when qemail is excluded")

    monkeypatch.setattr(report_module, "_process_removal_model", _raise_if_called)
    payload = _build_recommendations(report)
    assert payload["status"] == "ok"
    assert payload["items"]
    assert calls["count"] == 0


def test_known_lane_skips_qemail_model_when_qemail_is_excluded(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_EXCLUDED_PROCESSES", "qemail")
    report = {
        "known_issues": {
            "expected_findings": [
                {
                    "title": "Known qemail issue landmark",
                    "plugin_id": "analysis_close_cycle_contention",
                    "kind": "close_cycle_contention",
                    "where": {"process": "qemail"},
                    "min_count": 1,
                    "max_count": 1,
                }
            ]
        },
        "plugins": {
            "analysis_close_cycle_contention": {"findings": []},
        },
    }
    calls = {"count": 0}

    def _raise_if_called(*_args, **_kwargs):
        calls["count"] += 1
        raise AssertionError("qemail model should be skipped when qemail is excluded")

    monkeypatch.setattr(report_module, "_process_removal_model", _raise_if_called)
    payload = _build_recommendations(report)
    assert payload["status"] == "ok"
    assert isinstance(payload.get("known"), dict)
    assert calls["count"] == 0


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


def test_dependency_chain_flow_actions_backstop_to_process_local_action() -> None:
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
    assert items
    assert items[0].get("kind") == "plugin_actionability_backstop"
    assert items[0].get("action_type") == "batch_or_cache"


def test_chain_bound_flow_rewire_actions_are_filtered_but_process_local_actions_can_remain(
    monkeypatch,
) -> None:
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
                    {
                        "kind": "actionable_ops_lever",
                        "process_norm": "jbjeminhld",
                        "title": "Route dependent handoff",
                        "recommendation": "Route jbjeminhld handoff to a different chain.",
                        "action_type": "route_process",
                        "expected_delta_seconds": 3600.0,
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
    processes = [str(item.get("where", {}).get("process_norm") or "") for item in items]
    assert "rpt_por002" in processes
    assert "jbjeminhld" in processes
    assert "chain-bound" in str(payload.get("discovery", {}).get("summary") or "")


def test_discovery_recommendations_include_client_value_metrics(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_RANKING_VERSION", "v2")
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
                        "evidence": {
                            "estimated_calls_reduced": 200.0,
                            "top_user_run_share": 0.8,
                            "top_user_redacted": "user_a",
                            "distinct_users": 4,
                            "target_process_ids": ["rpt_por002"],
                            "metrics": {
                                "eligible_wait_p95_s": 120.0,
                                "modeled_wait_p95_s": 84.0,
                            },
                        },
                    }
                ]
            }
        }
    }
    payload = _build_recommendations(report)
    items = payload["items"]
    assert len(items) == 1
    row = items[0]
    assert row.get("scope_type") == "single_process"
    assert row.get("primary_process_id") == "rpt_por002"


def test_verified_route_action_plan_maps_to_discovery_recommendation(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_REQUIRE_MODELED_HOURS", "0")
    report = {
        "plugins": {
            "analysis_ebm_action_verifier_v1": {
                "findings": [
                    {
                        "id": "route-1",
                        "kind": "verified_route_action_plan",
                        "decision": "modeled",
                        "title": "Kona current→ideal route plan",
                        "total_delta_energy": 3.0,
                        "energy_before": 12.0,
                        "energy_after": 9.0,
                        "route_confidence": 0.84,
                        "measurement_type": "modeled",
                        "target": "rpt_por002",
                        "steps": [
                            {
                                "step_index": 1,
                                "action": "Convert payout extract to batch input mode.",
                                "confidence": 0.90,
                                "target_process_ids": ["rpt_por002"],
                            },
                            {
                                "step_index": 2,
                                "action": "Split oversized batches for payout extraction.",
                                "confidence": 0.78,
                                "target_process_ids": ["rpt_por002"],
                            },
                        ],
                        "evidence": {"route_plan_artifact": "artifacts/analysis_ebm_action_verifier_v1/route_plan.json"},
                    }
                ]
            }
        }
    }
    payload = _build_recommendations(report)
    rows = [r for r in payload.get("items") or [] if r.get("kind") == "verified_route_action_plan"]
    assert rows
    row = rows[0]
    assert row.get("action_type") == "route_process"
    assert "1. Convert payout extract to batch input mode." in str(row.get("recommendation") or "")
    assert "2. Split oversized batches for payout extraction." in str(row.get("recommendation") or "")
    assert float(row.get("modeled_percent_hint") or 0.0) > 0.0
    assert row.get("target_process_ids") == ["rpt_por002"]
    assert float(row.get("modeled_user_touches_reduced") or 0.0) > 0.0
    assert float(row.get("modeled_user_hours_saved_month") or 0.0) > 0.0
    assert float(row.get("modeled_close_hours_saved_month") or 0.0) > 0.0
    assert float(row.get("modeled_contention_reduction_pct_close") or 0.0) > 0.0
    assert float(row.get("value_score_v2") or 0.0) > 0.0
    assert row.get("optimization_metric") == "modeled_user_hours_saved"
    assert row.get("ranking_metric") == "client_value_score_v2"


def test_direct_action_type_aliases_are_normalized() -> None:
    report = {
        "plugins": {
            "analysis_actionable_ops_levers_v1": {
                "findings": [
                    {
                        "kind": "actionable_ops_lever",
                        "process_norm": "rpt_por002",
                        "title": "Simplify fixed parameter rule",
                        "recommendation": "Collapse repeated parameter rule variants.",
                        "action_type": "param_rule_simplification",
                        "expected_delta_seconds": 1800.0,
                        "confidence": 0.8,
                        "measurement_type": "modeled",
                    }
                ]
            }
        }
    }
    payload = _build_recommendations(report)
    items = payload["items"]
    assert len(items) == 1
    assert items[0].get("action_type") == "batch_input_refactor"


def test_v2_ranking_prefers_single_process_over_grouped(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_RANKING_VERSION", "v2")
    report = {
        "plugins": {
            "analysis_actionable_ops_levers_v1": {
                "findings": [
                    {
                        "kind": "actionable_ops_lever",
                        "process_norm": "rpt_por002",
                        "title": "Batch payout group",
                        "recommendation": "Convert payout chain to grouped batch inputs.",
                        "action_type": "batch_group_candidate",
                        "expected_delta_seconds": 7200.0,
                        "confidence": 0.85,
                        "measurement_type": "modeled",
                        "evidence": {
                            "target_process_ids": ["rpt_por002", "poextrprvn"],
                            "top_user_run_share": 0.8,
                            "estimated_calls_reduced": 160.0,
                        },
                    },
                    {
                        "kind": "actionable_ops_lever",
                        "process_norm": "rpt_por002",
                        "title": "Batch payout single",
                        "recommendation": "Convert process_id `rpt_por002` to single-process batch input.",
                        "action_type": "batch_input",
                        "expected_delta_seconds": 7200.0,
                        "confidence": 0.85,
                        "measurement_type": "modeled",
                        "evidence": {
                            "target_process_ids": ["rpt_por002"],
                            "top_user_run_share": 0.8,
                            "estimated_calls_reduced": 160.0,
                        },
                    },
                ]
            }
        }
    }
    payload = _build_recommendations(report)
    items = payload["items"]
    assert len(items) >= 2
    assert items[0].get("scope_type") == "single_process"
    assert items[0].get("action_type") == "batch_input"


def test_add_server_recommendation_is_routed_and_classified() -> None:
    report = {
        "plugins": {
            "analysis_queue_delay_decomposition": {
                "findings": [
                    {
                        "kind": "eligible_wait_impact",
                        "eligible_wait_hours_total": 90.0,
                    }
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
                        "delta_value": 33.3,
                        "unit": "percent",
                        "confidence": 0.8,
                        "measurement_type": "modeled",
                    }
                ]
            },
        }
    }
    payload = _build_recommendations(report)
    items = payload["items"]
    assert items
    add_server_rows = [row for row in items if str(row.get("action_type") or "").strip().lower() == "add_server"]
    assert add_server_rows
    row = add_server_rows[0]
    assert row.get("opportunity_class") == "server_capacity"
    assert float(row.get("modeled_efficiency_gain_pct") or 0.0) > 0.0


def test_verified_action_qpec_target_is_normalized_to_process(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_REQUIRE_DIRECT_PROCESS_ACTION", "1")
    monkeypatch.setenv("STAT_HARNESS_REQUIRE_MODELED_HOURS", "0")
    report = {
        "plugins": {
            "analysis_ebm_action_verifier_v1": {
                "findings": [
                    {
                        "kind": "verified_action",
                        "lever_id": "add_qpec_capacity_plus_one_v1",
                        "title": "Add QPEC capacity (+1) to reduce queue delays",
                        "what": "Add one QPEC worker under sustained load.",
                        "target": "LOCAL_MACHINE_ID=hash:abc,LOCAL_MACHINE_ID=hash:def",
                        "delta_energy": 10.0,
                        "energy_before": 100.0,
                        "energy_after": 90.0,
                        "measurement_type": "modeled",
                    }
                ]
            }
        }
    }
    payload = _build_recommendations(report)
    items = payload["items"]
    assert items
    routed = [item for item in items if item.get("plugin_id") == "analysis_ebm_action_verifier_v1"]
    assert routed
    assert routed[0].get("action_type") == "add_server"
    assert routed[0].get("where", {}).get("process_norm") == "qpec"


def test_param_variant_explosion_routes_to_batch_input_refactor() -> None:
    report = {
        "plugins": {
            "analysis_param_variant_explosion_v1": {
                "findings": [
                    {
                        "kind": "param_variant_explosion",
                        "measurement_type": "measured",
                        "recommendations": [
                            "Group and batch equivalent parameter variants where safe."
                        ],
                        "evidence": {
                            "metrics": {
                                "process": "rpt_por002",
                                "runs": 400.0,
                                "unique_params": 400.0,
                                "unique_ratio": 1.0,
                            }
                        },
                    }
                ]
            }
        }
    }
    payload = _build_recommendations(report)
    items = payload["items"]
    assert items
    routed = [item for item in items if item.get("plugin_id") == "analysis_param_variant_explosion_v1"]
    assert routed
    assert routed[0].get("action_type") == "batch_input_refactor"
    assert routed[0].get("where", {}).get("process_norm") == "rpt_por002"


def test_dynamic_close_detection_routes_indicator_processes_to_schedule_actions() -> None:
    report = {
        "plugins": {
            "analysis_dynamic_close_detection": {
                "findings": [
                    {
                        "kind": "close_cycle_roll",
                        "indicator_window_hours": 48.0,
                        "close_window_days": 10.0,
                        "close_end_delta_days": 2.0,
                        "indicator_processes": [
                            {"process": "jepresum", "share": 0.45, "count": 120},
                            {"process": "docopy", "share": 0.25, "count": 75},
                        ],
                    }
                ]
            }
        }
    }
    payload = _build_recommendations(report)
    items = payload["items"]
    routed = [item for item in items if item.get("plugin_id") == "analysis_dynamic_close_detection"]
    assert routed
    assert routed[0].get("action_type") == "tune_schedule"
    assert routed[0].get("where", {}).get("process_norm") in {"jepresum", "docopy"}
    assert float(routed[0].get("modeled_delta_hours") or 0.0) > 0.0


def test_actionability_coverage_counts_candidates_before_top_n_truncation() -> None:
    report = {
        "known_issues": {
            "recommendation_controls": {"top_n": 1},
        },
        "plugins": {
            "analysis_actionable_ops_levers_v1": {
                "findings": [
                    {
                        "kind": "actionable_ops_lever",
                        "process_norm": "rpt_por002",
                        "title": "Batch payout input",
                        "recommendation": "Convert process rpt_por002 to batch input.",
                        "action_type": "batch_input",
                        "expected_delta_seconds": 7200.0,
                        "confidence": 0.9,
                        "measurement_type": "modeled",
                    }
                ]
            },
            "analysis_param_variant_explosion_v1": {
                "findings": [
                    {
                        "kind": "param_variant_explosion",
                        "measurement_type": "measured",
                        "recommendations": [
                            "Batch equivalent variants for process pognrtrpt."
                        ],
                        "evidence": {
                            "metrics": {
                                "process": "pognrtrpt",
                                "runs": 240.0,
                                "unique_params": 180.0,
                                "unique_ratio": 0.75,
                            }
                        },
                    }
                ]
            },
        },
    }
    payload = _build_recommendations(report)
    items = payload.get("items") or []
    assert len(items) == 1
    explanations = payload.get("explanations") if isinstance(payload.get("explanations"), dict) else {}
    explanation_items = explanations.get("items") if isinstance(explanations.get("items"), list) else []
    explained_plugin_ids = {str(row.get("plugin_id") or "") for row in explanation_items if isinstance(row, dict)}
    assert "analysis_param_variant_explosion_v1" not in explained_plugin_ids
    coverage = payload.get("actionability_coverage") if isinstance(payload.get("actionability_coverage"), dict) else {}
    assert int(coverage.get("actionable_plugin_count") or 0) >= 2


def test_actionable_ops_can_resolve_process_from_queue_id_tokens(monkeypatch) -> None:
    monkeypatch.setattr(
        report_module,
        "_resolve_processes_from_queue_ids",
        lambda _report, _storage, _queue_ids: ["rpt_por002"],
    )
    report = {
        "plugins": {
            "analysis_frequent_itemsets_fpgrowth_v1": {
                "findings": [
                    {
                        "kind": "actionable_ops_lever",
                        "title": "Create a preset job for a frequent parameter bundle",
                        "recommendation": (
                            "These keys co-occur: process queue id(18065), t(18003). "
                            "Consider a single preset job/API."
                        ),
                        "process": "(multiple)",
                        "process_norm": "(multiple)",
                        "process_id": "proc:(multiple)",
                        "action_type": "preset_job_candidate",
                        "expected_delta_seconds": 3600.0,
                        "confidence": 0.8,
                        "measurement_type": "modeled",
                    }
                ]
            }
        }
    }
    payload = _build_recommendations(report)
    items = payload["items"]
    assert items
    row = items[0]
    assert row.get("where", {}).get("process_norm") == "rpt_por002"
    assert row.get("action_type") == "batch_input_refactor"


def test_actionable_ops_multiple_falls_back_to_variant_process() -> None:
    report = {
        "plugins": {
            "analysis_param_variant_explosion_v1": {
                "findings": [
                    {
                        "kind": "param_variant_explosion",
                        "measurement_type": "measured",
                        "evidence": {
                            "metrics": {
                                "process": "rpt_por002",
                                "runs": 400.0,
                                "unique_ratio": 0.9,
                            }
                        },
                    }
                ]
            },
            "analysis_association_rules_apriori_v1": {
                "findings": [
                    {
                        "kind": "actionable_ops_lever",
                        "title": "Simplify variants using a consistent parameter rule",
                        "recommendation": "When key A appears, key B appears. Collapse variants.",
                        "process_norm": "(multiple)",
                        "process": "(multiple)",
                        "process_id": "proc:(multiple)",
                        "action_type": "param_rule_simplification",
                        "expected_delta_seconds": 1800.0,
                        "confidence": 0.8,
                        "measurement_type": "modeled",
                    }
                ]
            },
        }
    }
    payload = _build_recommendations(report)
    items = [row for row in payload["items"] if row.get("plugin_id") == "analysis_association_rules_apriori_v1"]
    assert items
    assert items[0].get("where", {}).get("process_norm") == "rpt_por002"
    assert items[0].get("action_type") == "batch_input_refactor"


def test_close_cycle_change_point_routes_to_close_indicator_process() -> None:
    report = {
        "plugins": {
            "analysis_dynamic_close_detection": {
                "findings": [
                    {
                        "kind": "close_cycle_roll",
                        "indicator_processes": [
                            {"process": "jepresum", "share": 0.4, "count": 50},
                            {"process": "jeprecore", "share": 0.2, "count": 20},
                        ],
                    }
                ]
            },
            "analysis_close_cycle_change_point_v1": {
                "findings": [
                    {
                        "kind": "close_cycle_change_point",
                        "measurement_type": "modeled",
                        "evidence": {"metrics": {"ratio": 1.6, "days": 30}},
                        "recommendations": ["Investigate drift window."],
                    }
                ]
            },
        }
    }
    payload = _build_recommendations(report)
    rows = [row for row in payload["items"] if row.get("plugin_id") == "analysis_close_cycle_change_point_v1"]
    assert rows
    assert rows[0].get("action_type") == "tune_schedule"
    assert rows[0].get("where", {}).get("process_norm") in {"jepresum", "jeprecore"}


def test_leftfield_signal_routes_to_fallback_process() -> None:
    report = {
        "plugins": {
            "analysis_param_variant_explosion_v1": {
                "findings": [
                    {
                        "kind": "param_variant_explosion",
                        "measurement_type": "measured",
                        "evidence": {
                            "metrics": {
                                "process": "rpt_por002",
                                "runs": 250.0,
                                "unique_ratio": 0.8,
                            }
                        },
                    }
                ]
            },
            "analysis_cur_decomposition_explain_v1": {
                "findings": [
                    {
                        "kind": "leftfield_signal",
                        "title": "CUR identified influential columns",
                        "score": 0.25,
                        "recommendation": "Prioritize high-leverage fields.",
                        "evidence": {"column": "EXECUTE_AFTER_IND", "score": 0.25},
                    }
                ]
            },
        }
    }
    payload = _build_recommendations(report)
    rows = [row for row in payload["items"] if row.get("plugin_id") == "analysis_cur_decomposition_explain_v1"]
    assert rows
    assert rows[0].get("where", {}).get("process_norm") == "rpt_por002"
    assert rows[0].get("action_type") == "batch_input_refactor"


def test_action_cap_preserves_plugin_diversity_with_one_slot_override(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_MAX_PER_ACTION_TYPE", "batch_input_refactor=1")
    report = {
        "plugins": {
            "analysis_param_variant_explosion_v1": {
                "findings": [
                    {
                        "kind": "param_variant_explosion",
                        "measurement_type": "measured",
                        "evidence": {
                            "metrics": {
                                "process": "rpt_por002",
                                "runs": 400.0,
                                "unique_ratio": 1.0,
                            }
                        },
                    }
                ]
            },
            "analysis_association_rules_apriori_v1": {
                "findings": [
                    {
                        "kind": "actionable_ops_lever",
                        "title": "Simplify variants with a shared parameter rule",
                        "recommendation": "Collapse repeated parameter variants.",
                        "process_norm": "(multiple)",
                        "process": "(multiple)",
                        "process_id": "proc:(multiple)",
                        "action_type": "param_rule_simplification",
                        "expected_delta_seconds": 600.0,
                        "confidence": 0.8,
                        "measurement_type": "modeled",
                    }
                ]
            },
        }
    }
    payload = _build_recommendations(report)
    items = payload.get("items") or []
    plugin_ids = {str(row.get("plugin_id") or "") for row in items if isinstance(row, dict)}
    assert "analysis_param_variant_explosion_v1" in plugin_ids
    assert "analysis_association_rules_apriori_v1" in plugin_ids


def test_actionable_ops_without_explicit_delta_gets_modeled_fallback() -> None:
    report = {
        "plugins": {
            "analysis_param_variant_explosion_v1": {
                "findings": [
                    {
                        "kind": "param_variant_explosion",
                        "measurement_type": "measured",
                        "evidence": {
                            "metrics": {
                                "process": "rpt_por002",
                                "runs": 500.0,
                                "unique_ratio": 1.0,
                            }
                        },
                    }
                ]
            },
            "analysis_association_rules_apriori_v1": {
                "findings": [
                    {
                        "kind": "actionable_ops_lever",
                        "title": "Simplify variants with a shared parameter rule",
                        "recommendation": "Collapse repeated parameter variants.",
                        "process_norm": "(multiple)",
                        "process": "(multiple)",
                        "process_id": "proc:(multiple)",
                        "action_type": "param_rule_simplification",
                        "confidence": 0.8,
                        "measurement_type": "modeled",
                    }
                ]
            },
        }
    }
    payload = _build_recommendations(report)
    rows = [row for row in payload.get("items") or [] if row.get("plugin_id") == "analysis_association_rules_apriori_v1"]
    assert rows
    assert float(rows[0].get("modeled_delta_hours") or 0.0) > 0.0


def test_simulate_plan_alias_routes_to_batch_action_with_fallback_process() -> None:
    report = {
        "plugins": {
            "analysis_param_variant_explosion_v1": {
                "findings": [
                    {
                        "kind": "param_variant_explosion",
                        "measurement_type": "measured",
                        "evidence": {
                            "metrics": {
                                "process": "rpt_por002",
                                "runs": 120.0,
                                "unique_ratio": 0.9,
                            }
                        },
                    }
                ]
            },
            "analysis_discrete_event_queue_simulator_v1": {
                "findings": [
                    {
                        "kind": "actionable_ops_lever",
                        "title": "Simulation sanity-check for action plan",
                        "recommendation": "Run simulation plan for queue pressure reduction.",
                        "process_norm": "(multiple)",
                        "process": "(multiple)",
                        "process_id": "proc:(multiple)",
                        "action_type": "simulate_plan",
                        "confidence": 0.7,
                        "measurement_type": "modeled",
                    }
                ]
            },
        }
    }
    payload = _build_recommendations(report)
    rows = [
        row
        for row in payload.get("items") or []
        if row.get("plugin_id") == "analysis_discrete_event_queue_simulator_v1"
    ]
    assert rows
    assert rows[0].get("action_type") == "batch_or_cache"
    assert rows[0].get("where", {}).get("process_norm") == "rpt_por002"


def test_generic_analysis_kind_adapter_routes_anomaly_with_fallback_process() -> None:
    report = {
        "plugins": {
            "analysis_param_variant_explosion_v1": {
                "findings": [
                    {
                        "kind": "param_variant_explosion",
                        "measurement_type": "measured",
                        "evidence": {
                            "metrics": {
                                "process": "rpt_por002",
                                "runs": 300.0,
                                "unique_ratio": 1.0,
                            }
                        },
                    }
                ]
            },
            "analysis_conformal_feature_prediction": {
                "findings": [
                    {
                        "kind": "anomaly",
                        "title": "Conformal anomaly candidates detected",
                        "recommendation": "Anomaly score outliers detected in this slice.",
                        "score": 1.8,
                        "measurement_type": "modeled",
                    }
                ]
            },
        }
    }
    payload = _build_recommendations(report)
    rows = [
        row
        for row in payload.get("items") or []
        if row.get("plugin_id") == "analysis_conformal_feature_prediction"
    ]
    assert rows
    assert rows[0].get("action_type") == "batch_or_cache"
    assert rows[0].get("where", {}).get("process_norm") == "rpt_por002"
    assert float(rows[0].get("modeled_delta_hours") or 0.0) > 0.0


def test_post_filter_backstop_keeps_plugin_actionable_after_policy_drop() -> None:
    report = {
        "plugins": {
            "analysis_actionable_ops_levers_v1": {
                "findings": [
                    {
                        "kind": "actionable_ops_lever",
                        "process_norm": "jboachild",
                        "title": "Route handoff to alternate chain",
                        "recommendation": "Legacy flow rewire recommendation.",
                        "action_type": "route_process",
                        "expected_delta_seconds": 3600.0,
                        "confidence": 0.8,
                        "measurement_type": "modeled",
                    }
                ]
            }
        }
    }
    payload = _build_recommendations(report)
    rows = [
        row
        for row in payload.get("items") or []
        if row.get("plugin_id") == "analysis_actionable_ops_levers_v1"
    ]
    assert rows
    assert rows[0].get("kind") == "plugin_actionability_backstop"
    assert rows[0].get("action_type") == "batch_or_cache"
    assert rows[0].get("scope_class") in {"general", "close_specific"}
    assert "modeled_delta_hours" in rows[0]
    assert "modeled_percent" in rows[0]
    assert "modeled_basis_hours" in rows[0]
    assert "not_modeled_reason" in rows[0]
    coverage = payload.get("actionability_coverage") if isinstance(payload.get("actionability_coverage"), dict) else {}
    assert int(coverage.get("actionable_plugin_count") or 0) >= 1


def test_modeled_percent_is_clamped_to_schema_range() -> None:
    report = {
        "plugins": {
            "analysis_actionable_ops_levers_v1": {
                "findings": [
                    {
                        "kind": "actionable_ops_lever",
                        "process_norm": "rpt_por002",
                        "title": "Convert rpt_por002 to batched input",
                        "recommendation": "Convert rpt_por002 to multi-parameter batch input.",
                        "action_type": "batch_input",
                        "expected_delta_seconds": 3600.0,
                        "modeled_percent_hint": 256.41159424,
                        "measurement_type": "modeled",
                    }
                ]
            }
        }
    }
    payload = _build_recommendations(report)
    rows = [
        row
        for row in payload.get("items") or []
        if row.get("plugin_id") == "analysis_actionable_ops_levers_v1"
    ]
    assert rows
    modeled_pct = rows[0].get("modeled_percent")
    assert isinstance(modeled_pct, (int, float))
    assert 0.0 <= float(modeled_pct) <= 100.0
