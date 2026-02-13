from __future__ import annotations

import json
from pathlib import Path

from jsonschema import validate

from scripts.map_repo_improvements_to_capabilities import build_payload as build_map_payload
from scripts.normalize_repo_improvements_catalog import build_payload as build_normalized_payload
from scripts.plan_repo_improvements_rollout import (
    build_markdown as build_execution_markdown,
    build_payload as build_execution_payload,
)
from scripts.run_repo_improvements_pipeline import (
    build_status_markdown,
    build_status_payload,
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_repo_improvements_normalized_catalog_is_valid_and_up_to_date() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_payload = _read_json(root / "docs" / "repo_improvements_catalog_v3.json")
    canonical_payload = _read_json(root / "docs" / "repo_improvements_catalog_v3.canonical.json")
    touchpoint_map = _read_json(root / "docs" / "repo_improvements_touchpoint_map.json")
    expected = build_normalized_payload(
        raw_payload=raw_payload,
        canonical_payload=canonical_payload,
        touchpoint_map=touchpoint_map,
    )
    schema = _read_json(root / "docs" / "repo_improvements_catalog.normalized.schema.json")
    validate(instance=expected, schema=schema)
    existing = _read_json(root / "docs" / "repo_improvements_catalog_v3.normalized.json")
    assert existing == expected
    for item in existing.get("catalog") or []:
        for rel in item.get("normalized_touchpoints") or []:
            assert (root / rel).exists(), rel


def test_repo_improvements_capability_map_is_up_to_date() -> None:
    root = Path(__file__).resolve().parents[1]
    normalized_payload = _read_json(root / "docs" / "repo_improvements_catalog_v3.normalized.json")
    expected = build_map_payload(normalized_payload)
    existing = _read_json(root / "docs" / "repo_improvements_capability_map_v1.json")
    assert existing == expected


def test_repo_improvements_execution_plan_is_up_to_date() -> None:
    root = Path(__file__).resolve().parents[1]
    normalized_payload = _read_json(root / "docs" / "repo_improvements_catalog_v3.normalized.json")
    map_payload = _read_json(root / "docs" / "repo_improvements_capability_map_v1.json")
    expected = build_execution_payload(normalized_payload, map_payload)
    expected_md = build_execution_markdown(expected)
    existing = _read_json(root / "docs" / "repo_improvements_execution_plan_v1.json")
    existing_md = (root / "docs" / "repo_improvements_execution_plan_v1.md").read_text(
        encoding="utf-8"
    )
    assert existing == expected
    assert existing_md == expected_md


def test_repo_improvements_status_ledger_is_up_to_date() -> None:
    root = Path(__file__).resolve().parents[1]
    execution_plan = _read_json(root / "docs" / "repo_improvements_execution_plan_v1.json")
    expected = build_status_payload(execution_plan)
    expected_md = build_status_markdown(expected)
    existing = _read_json(root / "docs" / "repo_improvements_status.json")
    existing_md = (root / "docs" / "repo_improvements_status.md").read_text(encoding="utf-8")
    assert existing == expected
    assert existing_md == expected_md
