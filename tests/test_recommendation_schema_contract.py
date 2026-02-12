from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import ValidationError, validate


def _base_report_payload() -> dict:
    return {
        "run_id": "test-run",
        "created_at": "2026-02-12T00:00:00+00:00",
        "status": "ok",
        "lineage": {
            "run": {
                "run_id": "test-run",
                "created_at": "2026-02-12T00:00:00+00:00",
                "status": "ok",
                "run_seed": 42,
            },
            "input": {},
            "dataset": {"dataset_version_id": "dv-test"},
            "plugins": {},
        },
        "input": {"filename": "in.csv", "rows": 1, "cols": 1, "inferred_types": {}},
        "plugins": {},
        "recommendations": {
            "status": "ok",
            "summary": "demo",
            "known": {"status": "ok", "summary": "known", "items": []},
            "discovery": {"status": "ok", "summary": "discovery", "items": []},
            "items": [],
        },
    }


def _recommendation_item(**overrides: object) -> dict:
    base = {
        "status": "confirmed",
        "recommendation": "Do the thing.",
        "plugin_id": "analysis_actionable_ops_levers_v1",
        "kind": "actionable_ops_lever",
        "scope_class": "general",
        "modeled_percent": 12.5,
        "modeled_basis_hours": 100.0,
        "modeled_delta_hours": 12.5,
        "not_modeled_reason": None,
    }
    base.update(overrides)
    return base


def _load_schema() -> dict:
    return json.loads(Path("docs/report.schema.json").read_text(encoding="utf-8"))


def test_recommendation_schema_contract_accepts_modeled_item() -> None:
    report = _base_report_payload()
    item = _recommendation_item()
    report["recommendations"]["discovery"]["items"] = [item]
    report["recommendations"]["items"] = [item]
    validate(instance=report, schema=_load_schema())


def test_recommendation_schema_contract_requires_reason_when_not_modeled() -> None:
    report = _base_report_payload()
    item = _recommendation_item(modeled_percent=None, not_modeled_reason=None)
    report["recommendations"]["discovery"]["items"] = [item]
    report["recommendations"]["items"] = [item]
    with pytest.raises(ValidationError):
        validate(instance=report, schema=_load_schema())
