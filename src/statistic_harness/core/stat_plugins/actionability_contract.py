from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import ValidationError, validate


ROOT = Path(__file__).resolve().parents[4]
SCHEMA_PATH = ROOT / "docs" / "schemas" / "actionable_recommendation_contract_v2.json"


class ActionabilityContractError(ValueError):
    pass


def load_contract_schema() -> dict[str, Any]:
    payload = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ActionabilityContractError("Actionability contract schema must be an object")
    return payload


def validate_actionability_payload(payload: dict[str, Any]) -> None:
    try:
        validate(instance=payload, schema=load_contract_schema())
    except ValidationError as exc:
        raise ActionabilityContractError(str(exc)) from exc

