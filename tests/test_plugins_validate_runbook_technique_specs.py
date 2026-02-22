from __future__ import annotations

import json
from pathlib import Path


def test_plugins_validate_runbook_technique_specs_completeness() -> None:
    root = Path(".").resolve()
    map_path = root / "docs" / "release_evidence" / "plugins_validate_runbook_technique_map.json"
    rows = json.loads(map_path.read_text(encoding="utf-8"))
    specs_dir = root / "docs" / "release_evidence" / "plugins_validate_runbook_technique_specs"
    files = sorted(path for path in specs_dir.glob("*.md") if path.name != "README.md")
    assert len(rows) == 30
    assert len(files) == 30
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "Plugin ID:" in text
        assert "## Acceptance" in text
