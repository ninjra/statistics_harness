from __future__ import annotations

import json

from scripts.canonicalize_repo_improvements_catalog import (
    DEFAULT_CANONICAL_OUTPUT_PATH,
    DEFAULT_INPUT_PATH,
    DEFAULT_REDUCTION_OUTPUT_PATH,
    build_outputs_from_payload,
)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_canonicalization_is_deterministic() -> None:
    payload = _read_json(DEFAULT_INPUT_PATH)
    canonical_a, reduction_a = build_outputs_from_payload(payload)
    canonical_b, reduction_b = build_outputs_from_payload(payload)
    assert canonical_a == canonical_b
    assert reduction_a == reduction_b


def test_reduction_mapping_is_total_and_consistent() -> None:
    payload = _read_json(DEFAULT_INPUT_PATH)
    canonical, reduction = build_outputs_from_payload(payload)

    source_ids = sorted(str(item["id"]) for item in payload["catalog"])
    mapping = reduction["source_to_canonical"]
    assert sorted(mapping.keys()) == source_ids

    for entry in canonical["catalog"]:
        canonical_id = entry["canonical_item_id"]
        for source_id in entry["source_item_ids"]:
            assert mapping[source_id] == canonical_id


def test_generated_catalog_artifacts_are_up_to_date() -> None:
    payload = _read_json(DEFAULT_INPUT_PATH)
    expected_canonical, expected_reduction = build_outputs_from_payload(payload)
    existing_canonical = _read_json(DEFAULT_CANONICAL_OUTPUT_PATH)
    existing_reduction = _read_json(DEFAULT_REDUCTION_OUTPUT_PATH)
    assert existing_canonical == expected_canonical
    assert existing_reduction == expected_reduction
