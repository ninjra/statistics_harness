from __future__ import annotations

from scripts.run_loaded_dataset_full import _render_recommendations_plain_md


def test_plain_recommendations_includes_simple_sections() -> None:
    recs = {
        "summary": "demo",
        "discovery": {
            "items": [
                {
                    "plugin_id": "analysis_actionable_ops_levers_v1",
                    "kind": "actionable_ops_lever",
                    "action_type": "batch_group_candidate",
                    "where": {"process_norm": "rpt_por002"},
                    "recommendation": "technical text",
                    "impact_hours": 12.5,
                    "evidence": [
                        {
                            "target_process_ids": [
                                "rpt_por002",
                                "poextrprvn",
                                "poextrpexp",
                                "pognrtrpt",
                            ]
                        }
                    ],
                }
            ]
        },
        "known": {"items": []},
        "explanations": {"items": []},
    }
    known_checks = {"totals": {"total": 1, "confirmed": 1, "failing": 0}}
    out = _render_recommendations_plain_md(recs, known_checks)
    assert "What to change:" in out
    assert "Why this matters:" in out
    assert "Convert these process_ids to batch input first:" in out
    assert "Keep `rpt_por002` as a separate orchestration/report anchor recommendation." in out
    assert "rpt_por002" in out
    assert "Known-Issue Detection" in out


def test_plain_recommendations_surfaces_linked_manual_sweep_set() -> None:
    recs = {
        "summary": "demo",
        "discovery": {
            "items": [
                {
                    "plugin_id": "analysis_actionable_ops_levers_v1",
                    "kind": "actionable_ops_lever",
                    "action_type": "batch_input",
                    "where": {"process_norm": "gmicalcrun"},
                    "recommendation": "technical text",
                    "modeled_delta_hours": 0.33,
                    "modeled_user_touches_reduced": 923,
                    "affected_user_primary": "audrey_stachmus2",
                    "evidence": [{"key": "o(18015)", "top_user_redacted": "audrey_stachmus2", "close_month": "2025-08"}],
                },
                {
                    "plugin_id": "analysis_actionable_ops_levers_v1",
                    "kind": "actionable_ops_lever",
                    "action_type": "batch_input",
                    "where": {"process_norm": "domktgrpcr"},
                    "recommendation": "technical text",
                    "modeled_delta_hours": 0.32,
                    "modeled_user_touches_reduced": 910,
                    "affected_user_primary": "audrey_stachmus2",
                    "evidence": [{"key": "o(18015)", "top_user_redacted": "audrey_stachmus2", "close_month": "2025-08"}],
                },
            ]
        },
        "known": {"items": []},
        "explanations": {"items": []},
    }
    known_checks = {"totals": {"total": 1, "confirmed": 1, "failing": 0}}
    out = _render_recommendations_plain_md(recs, known_checks)
    assert "Linked recommendation set: gmicalcrun, domktgrpcr" in out
    assert "queue-delay hours understate this because it removes repetitive manual execution workload" in out
