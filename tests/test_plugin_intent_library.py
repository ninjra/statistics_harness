from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_plugin_intent_library import (
    DATA_ACCESS_MATRIX_PATH,
    FUNCTIONALITY_MATRIX_PATH,
    PLUGINS_ROOT,
    SQL_ASSIST_MATRIX_PATH,
    build_markdown,
    build_payload,
    build_rows,
)


def test_plugin_intent_library_generated_artifacts_are_up_to_date() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = build_rows(
        PLUGINS_ROOT,
        json.loads(FUNCTIONALITY_MATRIX_PATH.read_text(encoding="utf-8")),
        json.loads(DATA_ACCESS_MATRIX_PATH.read_text(encoding="utf-8")),
        json.loads(SQL_ASSIST_MATRIX_PATH.read_text(encoding="utf-8")),
    )
    expected_json = json.dumps(build_payload(rows), indent=2, sort_keys=True) + "\n"
    expected_md = build_markdown(rows)
    existing_json = (root / "docs" / "plugin_intent_library.json").read_text(encoding="utf-8")
    existing_md = (root / "docs" / "plugin_intent_library.md").read_text(encoding="utf-8")
    assert existing_json == expected_json
    assert existing_md == expected_md

