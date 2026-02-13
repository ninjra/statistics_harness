from __future__ import annotations

import json
from pathlib import Path

from scripts.sql_adoption_partition_matrix import generate_payload


def test_sql_adoption_partition_matrix_is_up_to_date_and_exclusive() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = generate_payload(root)

    groups = payload.get("groups") or {}
    grouped_ids = [pid for values in groups.values() for pid in values]
    candidate_ids = payload.get("candidate_plugin_ids") or []

    assert payload.get("candidate_count") == len(candidate_ids)
    assert sorted(grouped_ids) == sorted(candidate_ids)
    assert len(grouped_ids) == len(set(grouped_ids))
    assert payload.get("coverage", {}).get("is_complete") is True
    assert payload.get("coverage", {}).get("is_exclusive") is True

    existing = json.loads(
        (root / "docs" / "sql_adoption_partition_matrix.json").read_text(encoding="utf-8")
    )
    assert existing == payload
