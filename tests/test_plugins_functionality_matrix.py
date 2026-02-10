from __future__ import annotations

import json
from pathlib import Path

from scripts.plugins_functionality_matrix import build_matrix, _load_plugins


def test_plugins_functionality_matrix_is_up_to_date():
    root = Path(__file__).resolve().parents[1]
    plugins = _load_plugins(root / "plugins")
    matrix = build_matrix(plugins)

    # Enforce: checked-in matrix matches generator output.
    path = root / "docs" / "plugins_functionality_matrix.json"
    existing = json.loads(path.read_text(encoding="utf-8"))
    assert existing == matrix

