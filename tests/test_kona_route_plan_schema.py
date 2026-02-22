from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import ValidationError, validate


def _schema() -> dict:
    return json.loads(Path("docs/schemas/kona_route_plan.schema.json").read_text(encoding="utf-8"))


def test_kona_route_plan_schema_accepts_modeled_payload() -> None:
    payload = {
        "schema_version": "kona_route_plan.v1",
        "plugin_id": "analysis_ebm_action_verifier_v1",
        "generated_at": "2026-02-22T12:00:00Z",
        "decision": "modeled",
        "target_signature": "ALL",
        "config": {
            "route_max_depth": 2,
            "route_beam_width": 4,
            "route_min_delta_energy": 0.0,
            "route_min_confidence": 0.0,
            "route_allow_cross_target_steps": False,
            "route_stop_energy_threshold": 1.0,
            "route_candidate_limit": 10,
            "route_time_budget_ms": 0,
            "route_disallowed_lever_ids": [],
            "route_disallowed_action_types": [],
        },
        "not_applicable": None,
        "steps": [
            {
                "step_index": 1,
                "lever_id": "tune_schedule_qemail_frequency_v1",
                "action_type": "tune_schedule",
                "title": "Tune QEMAIL schedule",
                "action": "Increase QEMAIL interval.",
                "confidence": 0.9,
                "target_entity_keys": ["ALL"],
                "target_process_ids": ["qemail"],
                "energy_before": 10.0,
                "energy_after": 8.0,
                "delta_energy": 2.0,
                "modeled_metrics_after": {"queue_delay_p95": 90.0},
            }
        ],
        "totals": {
            "energy_before": 10.0,
            "energy_after": 8.0,
            "total_delta_energy": 2.0,
            "route_confidence": 0.9,
            "stop_reason": "max_depth",
            "expanded_states": 4,
        },
        "debug": {},
    }
    validate(instance=payload, schema=_schema())


def test_kona_route_plan_schema_accepts_not_applicable_payload() -> None:
    payload = {
        "schema_version": "kona_route_plan.v1",
        "plugin_id": "analysis_ebm_action_verifier_v1",
        "generated_at": "2026-02-22T12:00:00Z",
        "decision": "not_applicable",
        "target_signature": None,
        "config": {
            "route_max_depth": 0,
            "route_beam_width": 0,
            "route_min_delta_energy": 0.0,
            "route_min_confidence": 0.0,
            "route_allow_cross_target_steps": False,
            "route_stop_energy_threshold": 1.0,
            "route_candidate_limit": 10,
            "route_time_budget_ms": 0,
            "route_disallowed_lever_ids": [],
            "route_disallowed_action_types": [],
        },
        "not_applicable": {
            "reason_code": "ROUTE_DISABLED",
            "message": "Route planning is disabled by configuration.",
            "details": {},
        },
        "steps": [],
        "totals": {
            "energy_before": 0.0,
            "energy_after": 0.0,
            "total_delta_energy": 0.0,
            "route_confidence": 0.0,
            "stop_reason": "route_disabled",
            "expanded_states": 0,
        },
        "debug": {},
    }
    validate(instance=payload, schema=_schema())


def test_kona_route_plan_schema_rejects_modeled_without_steps() -> None:
    payload = {
        "schema_version": "kona_route_plan.v1",
        "plugin_id": "analysis_ebm_action_verifier_v1",
        "generated_at": "2026-02-22T12:00:00Z",
        "decision": "modeled",
        "target_signature": "ALL",
        "config": {
            "route_max_depth": 2,
            "route_beam_width": 4,
            "route_min_delta_energy": 0.0,
            "route_min_confidence": 0.0,
            "route_allow_cross_target_steps": False,
            "route_stop_energy_threshold": 1.0,
            "route_candidate_limit": 10,
            "route_time_budget_ms": 0,
            "route_disallowed_lever_ids": [],
            "route_disallowed_action_types": [],
        },
        "not_applicable": None,
        "steps": [],
        "totals": {
            "energy_before": 10.0,
            "energy_after": 8.0,
            "total_delta_energy": 2.0,
            "route_confidence": 0.9,
            "stop_reason": "max_depth",
            "expanded_states": 4,
        },
    }
    with pytest.raises(ValidationError):
        validate(instance=payload, schema=_schema())


def test_kona_route_plan_schema_rejects_not_applicable_missing_reason_fields() -> None:
    payload = {
        "schema_version": "kona_route_plan.v1",
        "plugin_id": "analysis_ebm_action_verifier_v1",
        "generated_at": "2026-02-22T12:00:00Z",
        "decision": "not_applicable",
        "target_signature": None,
        "config": {
            "route_max_depth": 0,
            "route_beam_width": 0,
            "route_min_delta_energy": 0.0,
            "route_min_confidence": 0.0,
            "route_allow_cross_target_steps": False,
            "route_stop_energy_threshold": 1.0,
            "route_candidate_limit": 10,
            "route_time_budget_ms": 0,
            "route_disallowed_lever_ids": [],
            "route_disallowed_action_types": [],
        },
        "not_applicable": {"details": {}},
        "steps": [],
        "totals": {
            "energy_before": 0.0,
            "energy_after": 0.0,
            "total_delta_energy": 0.0,
            "route_confidence": 0.0,
            "stop_reason": "route_disabled",
            "expanded_states": 0,
        },
    }
    with pytest.raises(ValidationError):
        validate(instance=payload, schema=_schema())
