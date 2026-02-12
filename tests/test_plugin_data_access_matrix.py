from __future__ import annotations

import json
from pathlib import Path

from scripts.plugin_data_access_matrix import ALLOWED_CONTRACTS, _as_json, generate


def test_plugin_data_access_matrix_is_up_to_date_and_classified() -> None:
    root = Path(__file__).resolve().parents[1]
    items = generate(root / "plugins")
    payload = _as_json(items)

    assert payload["plugin_count"] > 0
    assert payload["unclassified_count"] == 0

    for item in payload["plugins"]:
        contracts = item.get("access_contracts") or []
        assert contracts
        for contract in contracts:
            assert contract in ALLOWED_CONTRACTS

    existing = json.loads((root / "docs" / "plugin_data_access_matrix.json").read_text(encoding="utf-8"))
    assert existing == payload
