from __future__ import annotations

import json
from pathlib import Path

from scripts.top20_methods_matrix import as_markdown, build_matrix


def test_top20_methods_matrix_matches_checked_in_files() -> None:
    root = Path(__file__).resolve().parents[1]
    matrix_path = root / "docs" / "top20_additional_methods_plugins_matrix.json"
    md_path = root / "docs" / "top20_additional_methods_plugins_matrix.md"

    generated = build_matrix()
    assert int(generated.get("count") or 0) == 20

    existing = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert existing == generated
    assert md_path.read_text(encoding="utf-8").strip() == as_markdown(generated).strip()

