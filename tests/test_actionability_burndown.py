from __future__ import annotations

import json
from pathlib import Path

import scripts.actionability_burndown as burndown_mod


def test_build_payload_excludes_actionable(monkeypatch) -> None:
    def _fake_audit(_run_id: str, *, recompute: bool) -> dict[str, object]:
        assert recompute is False
        return {
            "plugins": [
                {
                    "plugin_id": "analysis_a",
                    "actionability_state": "actionable",
                    "reason_code": None,
                    "next_step_lane_id": "already_actionable",
                    "finding_count": 2,
                },
                {
                    "plugin_id": "analysis_b",
                    "actionability_state": "explained_non_actionable",
                    "reason_code": "NO_ACTIONABLE_RESULT",
                    "next_step_lane_id": "adapter_extension",
                    "finding_count": 1,
                    "recommended_next_step": "Add adapter",
                },
            ]
        }

    monkeypatch.setattr(burndown_mod, "audit_run", _fake_audit)
    payload = burndown_mod._build_payload("after_run", recompute=False)
    assert payload["run_id"] == "after_run"
    assert int(payload["unresolved_count"] or 0) == 1
    assert payload["reason_counts"] == {"NO_ACTIONABLE_RESULT": 1}
    assert payload["lane_counts"] == {"adapter_extension": 1}


def test_main_supports_before_run_delta(monkeypatch, tmp_path: Path) -> None:
    def _fake_audit(run_id: str, *, recompute: bool) -> dict[str, object]:
        assert recompute is False
        if run_id == "before_run":
            return {
                "plugins": [
                    {
                        "plugin_id": "analysis_x",
                        "actionability_state": "explained_non_actionable",
                        "reason_code": "NO_ACTIONABLE_RESULT",
                        "next_step_lane_id": "adapter_extension",
                        "finding_count": 1,
                        "recommended_next_step": "Add adapter",
                    }
                ]
            }
        return {
            "plugins": [
                {
                    "plugin_id": "analysis_x",
                    "actionability_state": "actionable",
                    "reason_code": None,
                    "next_step_lane_id": "already_actionable",
                    "finding_count": 1,
                },
                {
                    "plugin_id": "analysis_y",
                    "actionability_state": "explained_non_actionable",
                    "reason_code": "PREREQUISITE_UNMET",
                    "next_step_lane_id": "prerequisite_contract",
                    "finding_count": 2,
                    "recommended_next_step": "Verify prerequisites",
                },
            ]
        }

    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    monkeypatch.setattr(burndown_mod, "audit_run", _fake_audit)
    monkeypatch.setattr(
        "sys.argv",
        [
            "actionability_burndown.py",
            "--run-id",
            "after_run",
            "--before-run-id",
            "before_run",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
    )
    rc = burndown_mod.main()
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["before_run_id"] == "before_run"
    comparison = payload["comparison"]
    assert int(comparison["unresolved_count_before"] or 0) == 1
    assert int(comparison["unresolved_count_after"] or 0) == 1
    assert comparison["resolved_plugins"] == ["analysis_x"]
    assert comparison["newly_unresolved_plugins"] == ["analysis_y"]
