from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.show_actionable_results as sar


def _write_report(tmp_path: Path, run_id: str, report: dict[str, object]) -> None:
    run_dir = tmp_path / "appdata" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")


def test_top_n_selection_prefers_controllable_landmark(monkeypatch, tmp_path: Path, capsys) -> None:
    run_id = "r_unpinned"
    report = {
        "known_issues_mode": "off",
        "recommendations": {
            "items": [
                {
                    "plugin_id": "analysis_actionable_ops_levers_v1",
                    "kind": "actionable_ops_lever",
                    "primary_process_id": "abc123",
                    "action_type": "reschedule",
                    "recommendation": "Shift abc123 schedule.",
                    "impact_hours": 4.0,
                },
                {
                    "plugin_id": "analysis_capacity_scaling",
                    "kind": "capacity_scaling",
                    "primary_process_id": "qpec",
                    "action_type": "add_server",
                    "recommendation": "Add one qpec server.",
                    "impact_hours": 1.0,
                },
            ]
        },
        "plugins": {},
    }
    _write_report(tmp_path, run_id, report)
    monkeypatch.setattr(sar, "APPDATA", tmp_path / "appdata")
    monkeypatch.setattr(
        sys,
        "argv",
        ["show_actionable_results.py", "--run-id", run_id, "--top-n", "1", "--theme", "plain"],
    )

    rc = sar.main()
    out = capsys.readouterr().out

    assert rc == 0
    assert "| 1 | qpec |" in out
    assert "| 1 | abc123 |" not in out


def test_require_landmark_recall_fails_when_any_bucket_missing(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    run_id = "r_recall_fail"
    report = {
        "known_issues_mode": "on",
        "recommendations": {
            "items": [
                {
                    "plugin_id": "analysis_close_cycle_contention",
                    "kind": "actionable_ops_lever",
                    "primary_process_id": "qemail",
                    "action_type": "tune_schedule",
                    "recommendation": "Tune qemail schedule.",
                    "impact_hours": 3.0,
                },
                {
                    "plugin_id": "analysis_capacity_scaling",
                    "kind": "capacity_scaling",
                    "primary_process_id": "qpec",
                    "action_type": "add_server",
                    "recommendation": "Add one qpec server.",
                    "impact_hours": 2.0,
                },
            ]
        },
        "plugins": {},
    }
    _write_report(tmp_path, run_id, report)
    monkeypatch.setattr(sar, "APPDATA", tmp_path / "appdata")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "show_actionable_results.py",
            "--run-id",
            run_id,
            "--theme",
            "plain",
            "--require-landmark-recall",
            "--recall-top-n",
            "20",
        ],
    )

    rc = sar.main()
    out = capsys.readouterr().out

    assert rc == 2
    assert "recall_gate_status: FAIL" in out
    assert "payout_within_top_20" in out


def test_status_yn_contract() -> None:
    assert sar._status_yn("alpha", True) == "\x1b[32mY\x1b[0m:alpha"
    assert sar._status_yn("beta", False) == "\x1b[31mN\x1b[0m:beta"


def test_rank_uses_effort_and_time_not_plugin_family() -> None:
    items = [
        {
            "plugin_id": "analysis_actionable_ops_levers_v1",
            "kind": "actionable_ops_lever",
            "action_type": "batch_input",
            "relevance_score": 100.0,
            "impact_hours": 0.2,
            "delta_hours_close_dynamic": 0.2,
            "modeled_user_runs_reduced": 0.0,
        },
        {
            "plugin_id": "analysis_ideaspace_action_planner",
            "kind": "ideaspace_action",
            "action_type": "batch_input",
            "relevance_score": 1.0,
            "impact_hours": 10.0,
            "delta_hours_close_dynamic": 10.0,
            "modeled_user_runs_reduced": 0.0,
        },
    ]
    ranked = sar._ranked_actionables(items)
    assert ranked[0]["plugin_id"] == "analysis_ideaspace_action_planner"


def test_landmark_promotion_uses_ranked_items_without_synthetic_injection() -> None:
    ranked = [
        {
            "plugin_id": "analysis_generic",
            "kind": "verified_action",
            "primary_process_id": "abc123",
            "action_type": "reschedule",
            "recommendation": "Shift abc123.",
            "impact_hours": 9.0,
        },
        {
            "plugin_id": "analysis_close_cycle_contention",
            "kind": "verified_action",
            "primary_process_id": "qemail",
            "action_type": "tune_schedule",
            "recommendation": "Tune qemail schedule.",
            "impact_hours": 8.0,
        },
        {
            "plugin_id": "analysis_capacity_scaling",
            "kind": "capacity_scaling",
            "primary_process_id": "qpec",
            "action_type": "add_server",
            "recommendation": "Add one qpec server.",
            "impact_hours": 7.0,
        },
        {
            "plugin_id": "analysis_actionable_ops_levers_v1",
            "kind": "actionable_ops_lever",
            "primary_process_id": "rpt_por002",
            "action_type": "batch_group_candidate",
            "recommendation": "Convert payout chain to multi-input.",
            "impact_hours": 6.0,
        },
    ]
    selected = [ranked[0]]
    promoted, promoted_buckets = sar._promote_landmarks(selected, ranked, top_n=3)
    assert len(promoted) == 3
    assert "qemail" in promoted_buckets or "qpec" in promoted_buckets or "payout" in promoted_buckets
    assert any(str(item.get("primary_process_id") or "") == "qemail" for item in promoted)


def test_rank_prefers_user_effort_then_time() -> None:
    items = [
        {
            "plugin_id": "analysis_a",
            "kind": "actionable_ops_lever",
            "action_type": "batch_input",
            "modeled_user_runs_reduced": 20.0,
            "delta_hours_close_dynamic": 8.0,
            "relevance_score": 0.1,
        },
        {
            "plugin_id": "analysis_b",
            "kind": "actionable_ops_lever",
            "action_type": "batch_input",
            "modeled_user_runs_reduced": 300.0,
            "delta_hours_close_dynamic": 1.0,
            "relevance_score": 0.1,
        },
    ]
    ranked = sar._ranked_actionables(items)
    assert ranked[0]["plugin_id"] == "analysis_b"
