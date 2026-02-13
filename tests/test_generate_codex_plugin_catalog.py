from __future__ import annotations

from pathlib import Path

from scripts.generate_codex_plugin_catalog import build_markdown, build_rows


def test_generate_codex_plugin_catalog_is_up_to_date() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = build_markdown(build_rows(root / "plugins"))
    existing = (root / "docs" / "_codex_plugin_catalog.md").read_text(encoding="utf-8")
    assert existing == expected
