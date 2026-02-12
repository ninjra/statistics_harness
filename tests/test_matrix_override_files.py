from __future__ import annotations

import json
from pathlib import Path

from scripts.plugin_data_access_matrix import ALLOWED_CONTRACTS
from scripts.sql_assist_adoption_matrix import ALLOWED_INTENTS


def _plugin_ids(root: Path) -> set[str]:
    plugins_dir = root / "plugins"
    return {p.name for p in plugins_dir.iterdir() if p.is_dir()}


def test_plugin_data_access_overrides_are_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    plugin_ids = _plugin_ids(root)
    payload = json.loads((root / "docs" / "plugin_data_access_overrides.json").read_text(encoding="utf-8"))
    contracts = payload.get("contracts")
    assert isinstance(contracts, dict)
    for plugin_id, values in contracts.items():
        assert plugin_id in plugin_ids
        assert isinstance(values, list) and values
        for value in values:
            assert value in ALLOWED_CONTRACTS


def test_sql_assist_intent_overrides_are_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    plugin_ids = _plugin_ids(root)
    payload = json.loads((root / "docs" / "sql_assist_intent_overrides.json").read_text(encoding="utf-8"))
    intents = payload.get("intents")
    assert isinstance(intents, dict)
    for plugin_id, intent in intents.items():
        assert plugin_id in plugin_ids
        assert intent in ALLOWED_INTENTS
