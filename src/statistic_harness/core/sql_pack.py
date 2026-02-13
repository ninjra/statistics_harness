from __future__ import annotations

from pathlib import Path
from typing import Any

from jsonschema import validate

from statistic_harness.core.utils import read_json


def load_sql_pack(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("sql_pack must be a JSON object")
    return payload


def validate_sql_pack(payload: dict[str, Any], schema_path: Path) -> None:
    schema = read_json(schema_path)
    validate(instance=payload, schema=schema)

