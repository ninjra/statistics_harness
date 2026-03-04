#!/usr/bin/env python3
"""Audit which analysis plugins produce findings without a ``kind`` field.

Scans all 275 plugin directories and reports:
- Plugins whose plugin.py hard-codes ``kind`` in findings  (OK)
- Plugins that delegate to the registry (kind auto-filled by _enrich_findings)
- Plugins that neither set ``kind`` nor delegate  (NEEDS FIX)

Usage:
    python scripts/audit_plugin_finding_kinds.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PLUGINS_DIR = Path(__file__).resolve().parent.parent / "plugins"
REGISTRY_DELEGATES = re.compile(
    r"(registry\.run_plugin|run_plugin\(|from\s+statistic_harness\.core\.stat_plugins\s+import.*run_plugin)"
)
KIND_IN_FINDINGS = re.compile(r"""["']kind["']\s*:""")


def main() -> None:
    missing: list[str] = []
    has_kind: list[str] = []
    delegating: list[str] = []

    for plugin_dir in sorted(PLUGINS_DIR.iterdir()):
        if not plugin_dir.is_dir() or not plugin_dir.name.startswith("analysis_"):
            continue
        plugin_py = plugin_dir / "plugin.py"
        if not plugin_py.exists():
            missing.append(f"{plugin_dir.name} (no plugin.py)")
            continue

        source = plugin_py.read_text(encoding="utf-8", errors="replace")

        if REGISTRY_DELEGATES.search(source):
            delegating.append(plugin_dir.name)
            continue

        if KIND_IN_FINDINGS.search(source):
            has_kind.append(plugin_dir.name)
        else:
            missing.append(plugin_dir.name)

    print(f"=== Plugin Finding Kind Audit ===")
    print(f"  Delegates to registry (kind auto-filled): {len(delegating)}")
    print(f"  Hard-codes kind in findings:              {len(has_kind)}")
    print(f"  MISSING kind (needs fix):                 {len(missing)}")
    print()

    if missing:
        print("Plugins missing kind:")
        for name in sorted(missing):
            print(f"  - {name}")

    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
