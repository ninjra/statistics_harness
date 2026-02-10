from __future__ import annotations

import json
from pathlib import Path

from scripts.binding_implementation_matrix import as_json, scan_binding_docs


def test_binding_implementation_matrix_is_up_to_date_and_has_no_missing_refs():
    root = Path(__file__).resolve().parents[1]
    docs_root = root / "docs"
    plugins_root = root / "plugins"
    extra_docs = [root / "topo-tda-addon-pack-plan.md"]

    scans = scan_binding_docs(docs_root, plugins_root, extra_docs=extra_docs)
    payload = as_json(scans)

    assert payload["missing_any"] is False

    matrix_path = root / "docs" / "binding_implementation_matrix.json"
    existing = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert existing == payload

