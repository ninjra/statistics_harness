from __future__ import annotations

import json
from pathlib import Path

from scripts.docs_coverage_matrix import as_json, scan_docs


def test_docs_implementation_matrix_is_up_to_date_and_has_no_missing_refs():
    root = Path(__file__).resolve().parents[1]
    docs_root = root / "docs"
    plugins_root = root / "plugins"
    scans = scan_docs(docs_root, plugins_root)
    payload = as_json(scans)

    # Enforce: normative docs (excluding archived/generated) have no missing refs.
    assert payload["missing_any_normative"] is False

    # Enforce: checked-in matrix matches current generator output.
    matrix_path = root / "docs" / "implementation_matrix.json"
    existing = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert existing == payload
