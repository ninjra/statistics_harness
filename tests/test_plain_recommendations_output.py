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
    assert "rpt_por002" in out
    assert "Known-Issue Detection" in out
