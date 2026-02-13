from __future__ import annotations

import json
from pathlib import Path

from scripts.sql_assist_adoption_matrix import ALLOWED_INTENTS, _as_json, generate


def test_sql_assist_adoption_matrix_is_up_to_date_and_has_valid_intents() -> None:
    root = Path(__file__).resolve().parents[1]
    items = generate(root / "plugins")
    payload = _as_json(items)

    assert payload["plugin_count"] > 0

    for item in payload["plugins"]:
        assert item["sql_intent"] in ALLOWED_INTENTS

    existing = json.loads((root / "docs" / "sql_assist_adoption_matrix.json").read_text(encoding="utf-8"))
    assert existing == payload
