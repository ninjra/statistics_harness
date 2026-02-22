from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.build_plugins_validate_runbook_map import (
    TECHNIQUES,
    build_dependency_rows,
    build_payload,
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_plugins_validate_runbook_map_is_up_to_date() -> None:
    root = Path(".").resolve()
    expected = build_payload(root)
    existing = _read_json(
        root / "docs" / "release_evidence" / "plugins_validate_runbook_technique_map.json"
    )
    assert existing == expected
    assert len(existing) == len(TECHNIQUES) == 30
    assert len({row["ordinal"] for row in existing}) == 30
    assert len({row["technique"] for row in existing}) == 30


def test_plugins_validate_runbook_dependency_matrix_is_up_to_date() -> None:
    root = Path(".").resolve()
    mapping = build_payload(root)
    expected_rows = build_dependency_rows(root, mapping)
    csv_path = (
        root
        / "docs"
        / "release_evidence"
        / "plugins_validate_runbook_dependency_matrix.csv"
    )
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_rows = list(reader)
    normalized = []
    for row in expected_rows:
        normalized.append(
            {
                "ordinal": str(row["ordinal"]),
                "technique": row["technique"],
                "plugin_id": row["plugin_id"],
                "implemented": "Y" if row["implemented"] else "N",
                "depends_on": ";".join(row["depends_on"]),
                "consumer_count": str(row["consumer_count"]),
                "consumers": ";".join(row["consumers"]),
            }
        )
    assert actual_rows == normalized
