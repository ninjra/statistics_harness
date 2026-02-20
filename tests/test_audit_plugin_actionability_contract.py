from __future__ import annotations

import json

from scripts.audit_plugin_actionability import _build_next_step_work_contract
import scripts.audit_plugin_actionability as actionability_audit


def test_next_step_work_contract_classifies_known_lanes() -> None:
    rows = [
        {
            "plugin_id": "analysis_a",
            "actionability_state": "actionable",
            "reason_code": None,
            "recommended_next_step": None,
        },
        {
            "plugin_id": "analysis_b",
            "actionability_state": "explained_non_actionable",
            "reason_code": "OBSERVATION_ONLY",
            "recommended_next_step": "Add or extend recommendation adapters for this plugin's finding families.",
        },
        {
            "plugin_id": "analysis_c",
            "actionability_state": "explained_non_actionable",
            "reason_code": "NO_ACTIONABLE_FINDING_CLASS",
            "recommended_next_step": "Emit a direct-action finding kind (`actionable_ops_lever`, `ideaspace_action`, or `verified_action`) with process target and modeled delta fields.",
        },
        {
            "plugin_id": "analysis_d",
            "actionability_state": "explained_non_actionable",
            "reason_code": "NO_DIRECT_PROCESS_TARGET",
            "recommended_next_step": "Emit process-level targets (`process_norm` or `target_process_ids`) for each action candidate, then rerun full gauntlet.",
        },
        {
            "plugin_id": "report_x",
            "actionability_state": "explained_non_actionable",
            "reason_code": "REPORT_SNAPSHOT_OMISSION",
            "recommended_next_step": "Plugin executed but is missing from report.plugins snapshot; include it in report serialization.",
        },
    ]

    contract = _build_next_step_work_contract(rows)

    assert contract["plugin_count"] == 5
    assert contract["cluster_counts"]["already_actionable"] == 1
    assert contract["cluster_counts"]["adapter_extension"] == 1
    assert contract["cluster_counts"]["direct_action_contract"] == 1
    assert contract["cluster_counts"]["process_target_emission"] == 1
    assert contract["cluster_counts"]["report_snapshot_serialization"] == 1
    assert contract["unmapped_next_step_plugins"] == []
    assert contract["blank_non_actionable_next_step_plugins"] == []


def test_next_step_work_contract_flags_blank_non_actionable_step() -> None:
    rows = [
        {
            "plugin_id": "analysis_x",
            "actionability_state": "explained_non_actionable",
            "reason_code": "NO_ACTIONABLE_FINDING_CLASS",
            "recommended_next_step": "",
        }
    ]
    contract = _build_next_step_work_contract(rows)
    assert contract["unmapped_next_step_plugins"] == ["analysis_x"]
    assert contract["blank_non_actionable_next_step_plugins"] == ["analysis_x"]


def test_next_step_work_contract_flags_unmapped_next_step() -> None:
    rows = [
        {
            "plugin_id": "analysis_y",
            "actionability_state": "explained_non_actionable",
            "reason_code": "NO_ACTIONABLE_FINDING_CLASS",
            "recommended_next_step": "Do some custom unknown thing.",
        }
    ]
    contract = _build_next_step_work_contract(rows)
    assert contract["unmapped_next_step_plugins"] == ["analysis_y"]
    assert contract["blank_non_actionable_next_step_plugins"] == []


def test_audit_run_marks_discovery_actionable_plugin_even_without_rendered_item(
    tmp_path, monkeypatch
) -> None:
    run_id = "unit_run"
    run_dir = tmp_path / "appdata" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.json"
    report_path.write_text("{}", encoding="utf-8")

    report = {
        "plugins": {
            "analysis_a": {"status": "ok", "findings": [{"kind": "actionable_ops_lever"}]},
            "analysis_b": {"status": "ok", "findings": [{"kind": "param_variant_explosion"}]},
        }
    }
    recommendations = {
        "items": [{"plugin_id": "analysis_a", "action_type": "batch_input_refactor"}],
        "discovery": {"actionable_plugin_ids_all": ["analysis_a", "analysis_b"]},
        "explanations": {
            "items": [
                {
                    "plugin_id": "analysis_b",
                    "reason_code": "ADAPTER_RULE_MISSING",
                    "recommended_next_step": "Add or extend recommendation adapters.",
                }
            ]
        },
    }

    monkeypatch.setattr(actionability_audit, "ROOT", tmp_path)
    monkeypatch.setattr(actionability_audit, "_load_json", lambda path: json.loads(json.dumps(report)))
    monkeypatch.setattr(
        actionability_audit,
        "_recommendations_for_run",
        lambda report_payload, *, run_id, recompute: json.loads(json.dumps(recommendations)),
    )
    monkeypatch.setattr(
        actionability_audit,
        "_executed_plugin_rows",
        lambda run_id: {
            "analysis_a": {"execution_status": "ok"},
            "analysis_b": {"execution_status": "ok"},
        },
    )
    monkeypatch.setattr(
        actionability_audit,
        "_manifest_index",
        lambda: {
            "analysis_a": {"plugin_type": "analysis", "name": "analysis_a"},
            "analysis_b": {"plugin_type": "analysis", "name": "analysis_b"},
        },
    )

    payload = actionability_audit.audit_run(run_id, recompute=False)
    by_plugin = {
        str(row.get("plugin_id")): row
        for row in (payload.get("plugins") or [])
        if isinstance(row, dict)
    }
    assert by_plugin["analysis_a"]["actionability_state"] == "actionable"
    assert by_plugin["analysis_b"]["actionability_state"] == "actionable"
    assert by_plugin["analysis_b"]["recommendation_count"] == 0
    assert by_plugin["analysis_b"]["actionable_via_discovery"] is True
    assert by_plugin["analysis_b"]["reason_code"] is None
