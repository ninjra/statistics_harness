from __future__ import annotations

from pathlib import Path

from scripts.generate_codex_plugin_catalog import build_markdown, build_rows
from scripts.generate_codex_repo_manifest import build_manifest_lines


def test_codex_repo_manifest_is_up_to_date() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = "\n".join(build_manifest_lines()) + "\n"
    existing = (root / "docs" / "_codex_repo_manifest.txt").read_text(encoding="utf-8")
    assert existing == expected


def test_codex_plugin_catalog_is_up_to_date() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = build_rows(root / "plugins")
    expected = build_markdown(rows)
    existing = (root / "docs" / "_codex_plugin_catalog.md").read_text(encoding="utf-8")
    assert existing == expected
