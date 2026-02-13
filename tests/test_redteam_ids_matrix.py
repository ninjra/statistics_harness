from __future__ import annotations

import json
from pathlib import Path

from scripts.redteam_ids_matrix import build_matrix


def test_redteam_ids_matrix_is_up_to_date_and_has_no_missing_required_ids():
    root = Path(__file__).resolve().parents[1]
    matrix = build_matrix()

    assert matrix["missing_required_ids"] == []
    assert [x["id"] for x in matrix["items"] if x["status"] != "implemented"] == []

    path = root / "docs" / "redteam_ids_matrix.json"
    existing = json.loads(path.read_text(encoding="utf-8"))
    assert existing == matrix
