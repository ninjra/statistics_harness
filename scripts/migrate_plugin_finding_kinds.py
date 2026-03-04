#!/usr/bin/env python3
"""Batch-add ``kind`` field to full-implementation plugin findings.

Reads plugin_kind_map.yaml to determine the canonical kind for each plugin,
then patches plugin.py files that construct findings without a ``kind`` key.

Targets the pattern: findings.append({  or  {"id":  or  {"title":
and inserts "kind": "<canonical_kind>" as the first key in the dict.

Usage:
    python scripts/migrate_plugin_finding_kinds.py [--dry-run]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

PLUGINS_DIR = Path(__file__).resolve().parent.parent / "plugins"
KIND_MAP_PATH = Path(__file__).resolve().parent.parent / "config" / "plugin_kind_map.yaml"

# Patterns that indicate the plugin delegates to the registry
REGISTRY_DELEGATES = re.compile(
    r"(registry\.run_plugin|run_plugin\(|from\s+statistic_harness\.core\.stat_plugins\s+import.*run_plugin)"
)
# Check if kind is already present in findings
KIND_IN_FINDINGS = re.compile(r"""["']kind["']\s*:""")

# Pattern: findings.append({ or findings += [{ — first dict key after opening brace
# We'll insert "kind": "xxx", right after the opening { of finding dicts
FINDING_DICT_OPEN = re.compile(
    r'(findings\.append\(\s*\{|findings\s*\+=\s*\[\s*\{|findings\s*=\s*\[\s*\{|"findings"\s*:\s*\[\s*\{|\bfinding\s*=\s*\{)'
)


def migrate_plugin(plugin_dir: Path, kind: str, dry_run: bool) -> bool:
    """Add kind to finding dicts in plugin.py. Returns True if modified."""
    plugin_py = plugin_dir / "plugin.py"
    if not plugin_py.exists():
        return False

    source = plugin_py.read_text(encoding="utf-8")

    if REGISTRY_DELEGATES.search(source):
        return False  # Registry handles kind via _enrich_findings
    if KIND_IN_FINDINGS.search(source):
        return False  # Already has kind

    # Strategy: find all finding dict constructions and insert kind
    # Look for patterns like:  findings.append({  or  finding = {
    # and insert "kind": "xxx", right after the opening brace
    new_source = source
    modified = False

    # Pattern 1: findings.append({  → findings.append({"kind": "xxx",
    pattern1 = re.compile(r'(findings\.append\(\s*)\{(\s*\n)')
    if pattern1.search(new_source):
        new_source = pattern1.sub(rf'\g<1>{{"kind": "{kind}",\2', new_source)
        modified = True

    # Pattern 2: findings.append({\n            "id" → findings.append({\n            "kind": "xxx",\n            "id"
    if not modified:
        pattern2 = re.compile(r'(findings\.append\(\s*\{)\s*\n(\s+)("(?:id|title|severity|what)")')
        if pattern2.search(new_source):
            new_source = pattern2.sub(
                rf'\1\n\2"kind": "{kind}",\n\2\3',
                new_source,
            )
            modified = True

    # Pattern 3: finding = { or result_dict = {  followed by "id" or "title"
    if not modified:
        pattern3 = re.compile(r'((?:finding|result_dict|row|entry|item)\s*=\s*\{)\s*\n(\s+)("(?:id|title|severity|what)")')
        if pattern3.search(new_source):
            new_source = pattern3.sub(
                rf'\1\n\2"kind": "{kind}",\n\2\3',
                new_source,
            )
            modified = True

    # Pattern 4: return PluginResult(findings=[{  or findings=[{
    if not modified:
        pattern4 = re.compile(r'(findings\s*=\s*\[\s*\{)\s*\n(\s+)("(?:id|title|severity|what)")')
        if pattern4.search(new_source):
            new_source = pattern4.sub(
                rf'\1\n\2"kind": "{kind}",\n\2\3',
                new_source,
            )
            modified = True

    # Pattern 5: Generic dict literal with "id" as first key
    if not modified:
        pattern5 = re.compile(r'(\{\s*\n\s+)("id"\s*:\s*)')
        matches = list(pattern5.finditer(new_source))
        if matches:
            # Only modify if it looks like a finding dict (has "title" or "what" nearby)
            for m in reversed(matches):
                context = new_source[m.start():m.start() + 500]
                if '"title"' in context or '"what"' in context or '"severity"' in context:
                    indent = re.search(r'\n(\s+)', new_source[m.start():m.end()])
                    if indent:
                        ws = indent.group(1)
                        new_source = (
                            new_source[:m.start()]
                            + f'{{\n{ws}"kind": "{kind}",\n{ws}'
                            + new_source[m.start() + len(m.group(1)):]
                        )
                        modified = True

    if modified and not dry_run:
        plugin_py.write_text(new_source, encoding="utf-8")

    return modified


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    kind_map = yaml.safe_load(KIND_MAP_PATH.read_text(encoding="utf-8"))
    mappings = kind_map.get("mappings", {})

    migrated = []
    failed = []

    for plugin_dir in sorted(PLUGINS_DIR.iterdir()):
        if not plugin_dir.is_dir() or not plugin_dir.name.startswith("analysis_"):
            continue
        kind = mappings.get(plugin_dir.name)
        if not kind:
            continue

        result = migrate_plugin(plugin_dir, kind, dry_run)
        if result:
            migrated.append(plugin_dir.name)

    # Re-audit to find remaining
    for plugin_dir in sorted(PLUGINS_DIR.iterdir()):
        if not plugin_dir.is_dir() or not plugin_dir.name.startswith("analysis_"):
            continue
        plugin_py = plugin_dir / "plugin.py"
        if not plugin_py.exists():
            continue
        source = plugin_py.read_text(encoding="utf-8", errors="replace")
        if REGISTRY_DELEGATES.search(source):
            continue
        if KIND_IN_FINDINGS.search(source):
            continue
        failed.append(plugin_dir.name)

    prefix = "[DRY RUN] " if dry_run else ""
    print(f"{prefix}Migrated: {len(migrated)}")
    for name in migrated:
        print(f"  + {name}")
    print(f"{prefix}Still missing kind: {len(failed)}")
    for name in failed:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
