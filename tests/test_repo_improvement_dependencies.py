from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_repo_improvement_dependencies import validate_plan_payload


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_repo_improvement_dependency_validation_is_up_to_date_and_clean() -> None:
    root = Path(__file__).resolve().parents[1]
    plan = _read_json(root / "docs" / "repo_improvements_execution_plan_v1.json")
    expected = validate_plan_payload(plan)
    existing = _read_json(root / "docs" / "repo_improvements_dependency_validation_v1.json")
    assert expected["has_errors"] is False
    assert existing == expected


def test_repo_improvement_dependency_validator_detects_cycle() -> None:
    payload = {
        "items": [
            {"canonical_item_id": "CANONICAL_001", "dependency_ids": ["CANONICAL_002"]},
            {"canonical_item_id": "CANONICAL_002", "dependency_ids": ["CANONICAL_001"]},
        ]
    }
    out = validate_plan_payload(payload)
    assert out["has_errors"] is True
    assert out["cycles"]
