from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import validate

from statistic_harness.core.utils import read_json


def schema_sha256(schema_path: Path) -> str:
    payload = schema_path.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(obj: Any) -> str:
    # Deterministic JSON for hashing.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def signed_digest(payload: dict[str, Any]) -> str:
    material = dict(payload)
    material.pop("signature", None)
    return hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def load_signed_baseline(path: Path, schema_path: Path) -> dict[str, Any]:
    data = read_json(path)
    schema = read_json(schema_path)
    validate(instance=data, schema=schema)
    expected_schema_hash = schema_sha256(schema_path)
    if data.get("schema_hash") != expected_schema_hash:
        raise ValueError("baseline schema_hash mismatch")
    sig = data.get("signature") or {}
    if sig.get("algo") != "sha256":
        raise ValueError("baseline signature algo must be sha256")
    expected_digest = signed_digest(data)
    if sig.get("digest") != expected_digest:
        raise ValueError("baseline signature digest mismatch")
    return data

