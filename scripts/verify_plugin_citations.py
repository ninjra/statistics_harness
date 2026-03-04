#!/usr/bin/env python3
"""Verify all plugin.yaml files have a ``citation:`` block.

Checks that every plugin directory contains a plugin.yaml with at minimum:
- citation.method  (non-empty string)
- citation.finding_kind  (non-empty string)

Usage:
    python scripts/verify_plugin_citations.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

PLUGINS_DIR = Path(__file__).resolve().parent.parent / "plugins"


def main() -> None:
    total = 0
    valid = 0
    missing_citation: list[str] = []
    missing_method: list[str] = []
    missing_kind: list[str] = []

    for plugin_dir in sorted(PLUGINS_DIR.iterdir()):
        if not plugin_dir.is_dir():
            continue
        yaml_path = plugin_dir / "plugin.yaml"
        if not yaml_path.exists():
            continue

        total += 1
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except Exception:
            missing_citation.append(plugin_dir.name)
            continue

        if not isinstance(data, dict):
            missing_citation.append(plugin_dir.name)
            continue

        citation = data.get("citation")
        if not isinstance(citation, dict):
            missing_citation.append(plugin_dir.name)
            continue

        ok = True
        if not citation.get("method"):
            missing_method.append(plugin_dir.name)
            ok = False
        if not citation.get("finding_kind"):
            missing_kind.append(plugin_dir.name)
            ok = False
        if ok:
            valid += 1

    print(f"=== Plugin Citation Verification ===")
    print(f"  Total plugins: {total}")
    print(f"  Valid citations: {valid}")
    print(f"  Missing citation block: {len(missing_citation)}")
    print(f"  Missing method: {len(missing_method)}")
    print(f"  Missing finding_kind: {len(missing_kind)}")

    if missing_citation:
        print(f"\nPlugins missing citation:")
        for name in missing_citation:
            print(f"  - {name}")
    if missing_method:
        print(f"\nPlugins missing method:")
        for name in missing_method:
            print(f"  - {name}")
    if missing_kind:
        print(f"\nPlugins missing finding_kind:")
        for name in missing_kind:
            print(f"  - {name}")

    all_ok = not missing_citation and not missing_method and not missing_kind
    print(f"\nResult: {'PASS' if all_ok else 'FAIL'} ({valid}/{total})")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
