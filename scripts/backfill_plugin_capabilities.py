#!/usr/bin/env python3
"""Backfill correct capability tags into plugin.yaml files.

Reads each plugin's plugin.py source code, infers data-requirement tags
based on code patterns, and writes them to the plugin.yaml capabilities field.

Preserves existing valid tags (e.g. topo_tda_addon, diagnostic_only).
Removes the meaningless 'analysis' tag.

Usage:
    python scripts/backfill_plugin_capabilities.py [--dry-run]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

PLUGINS_DIR = Path("plugins")

# Tags that should be preserved if already present
PRESERVED_TAGS = {"topo_tda_addon", "diagnostic_only", "needs_coords", "has_coords", "has_point_id", "needs_point_id"}

# Tags to always remove (meaningless or wrong)
REMOVE_TAGS = {"analysis"}

# Inference rules: tag -> list of code patterns that indicate the tag is needed
CAPABILITY_INFERENCE_RULES: dict[str, list[str]] = {
    "needs_numeric": [
        "pd.to_numeric",
        ".astype(float",
        ".astype(int",
        "np.float",
        "np.nanmean",
        "np.nanstd",
        "np.mean(",
        "np.std(",
        "float(",
        ".select_dtypes",
        "numeric_cols",
        "numeric_columns",
        "is_numeric_dtype",
        "_numeric",
        "to_numeric",
    ],
    "needs_timestamp": [
        "pd.to_datetime",
        "to_datetime",
        "timestamp",
        "datetime",
        "parse_date",
        "time_col",
        "date_col",
        "close_date",
        "open_date",
        "created_at",
        "close_start",
    ],
    "needs_eventlog": [
        "process_column",
        "process_col",
        "activity",
        "case_id",
        "trace",
        "event_log",
        "eventlog",
        "process_id",
        "primary_process",
    ],
    "needs_host": [
        "host_column",
        "host_col",
        "host_id",
        "server_col",
        "server_id",
        "node_col",
        "node_id",
        "assigned_to",
    ],
}

# Minimum matches required before assigning a tag
# (prevents false positives from a single stray reference)
MIN_MATCHES: dict[str, int] = {
    "needs_numeric": 1,
    "needs_timestamp": 1,
    "needs_eventlog": 1,
    "needs_host": 1,
}


def infer_capabilities(source: str) -> set[str]:
    """Infer capability tags from plugin source code."""
    caps: set[str] = set()
    source_lower = source.lower()

    for tag, patterns in CAPABILITY_INFERENCE_RULES.items():
        match_count = sum(1 for p in patterns if p.lower() in source_lower)
        if match_count >= MIN_MATCHES.get(tag, 1):
            caps.add(tag)

    return caps


def load_yaml_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def update_capabilities_in_yaml(yaml_text: str, new_caps: list[str]) -> str:
    """Replace the capabilities block in YAML text, handling both inline and multi-line formats."""
    caps_str = "[" + ", ".join(new_caps) + "]" if new_caps else "[]"
    replacement = f"capabilities: {caps_str}"

    # Pattern 1: inline capabilities: [...] possibly followed by orphaned multi-line items
    pattern1 = re.compile(
        r"^capabilities:\s*\[.*?\]\n(?:\s*- \S+\n)*",
        re.MULTILINE,
    )
    new_text, count = pattern1.subn(replacement + "\n", yaml_text, count=1)
    if count > 0:
        return new_text

    # Pattern 2: multi-line capabilities block (capabilities:\n  - tag1\n  - tag2\n)
    pattern2 = re.compile(
        r"^capabilities:\s*\n(?:\s*- \S+\n)*",
        re.MULTILINE,
    )
    new_text, count = pattern2.subn(replacement + "\n", yaml_text, count=1)
    if count > 0:
        return new_text

    # Pattern 3: simple single-line (capabilities: something)
    pattern3 = re.compile(r"^capabilities:.*$", re.MULTILINE)
    new_text, count = pattern3.subn(replacement, yaml_text, count=1)

    return new_text


def process_plugin(plugin_dir: Path, dry_run: bool) -> dict:
    """Process a single plugin directory. Returns a summary dict."""
    yaml_path = plugin_dir / "plugin.yaml"
    py_path = plugin_dir / "plugin.py"

    if not yaml_path.exists():
        return {"plugin": plugin_dir.name, "status": "skip", "reason": "no plugin.yaml"}

    yaml_text = load_yaml_text(yaml_path)
    parsed = yaml.safe_load(yaml_text)
    if not isinstance(parsed, dict):
        return {"plugin": plugin_dir.name, "status": "skip", "reason": "invalid yaml"}

    plugin_type = str(parsed.get("type", ""))
    if plugin_type != "analysis":
        return {"plugin": plugin_dir.name, "status": "skip", "reason": f"type={plugin_type}"}

    # Get existing capabilities
    existing = set(parsed.get("capabilities") or [])
    preserved = existing & PRESERVED_TAGS

    # Infer from source
    inferred: set[str] = set()
    if py_path.exists():
        source = py_path.read_text(encoding="utf-8")
        inferred = infer_capabilities(source)

    # Build final set: preserved + inferred, minus removed
    final = (preserved | inferred) - REMOVE_TAGS
    final_sorted = sorted(final)

    old_sorted = sorted(existing - REMOVE_TAGS)
    changed = final_sorted != old_sorted or ("analysis" in existing)

    if changed and not dry_run:
        new_yaml = update_capabilities_in_yaml(yaml_text, final_sorted)
        yaml_path.write_text(new_yaml, encoding="utf-8")

    return {
        "plugin": plugin_dir.name,
        "status": "changed" if changed else "unchanged",
        "old": sorted(existing),
        "new": final_sorted,
        "inferred": sorted(inferred),
        "preserved": sorted(preserved),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args()

    results = []
    for plugin_dir in sorted(PLUGINS_DIR.iterdir()):
        if not plugin_dir.is_dir():
            continue
        result = process_plugin(plugin_dir, args.dry_run)
        results.append(result)

    changed = [r for r in results if r["status"] == "changed"]
    unchanged = [r for r in results if r["status"] == "unchanged"]
    skipped = [r for r in results if r["status"] == "skip"]

    print(f"Total plugins scanned: {len(results)}")
    print(f"Changed: {len(changed)}")
    print(f"Unchanged: {len(unchanged)}")
    print(f"Skipped (non-analysis): {len(skipped)}")

    if args.dry_run:
        print("\n--- DRY RUN (no files written) ---\n")

    # Show changes
    if changed:
        print("\nChanged plugins:")
        for r in changed:
            print(f"  {r['plugin']}: {r['old']} -> {r['new']}")

    # Tag distribution
    from collections import Counter
    tag_counts: Counter[str] = Counter()
    for r in results:
        if r["status"] != "skip":
            for tag in r.get("new", []):
                tag_counts[tag] += 1
    if tag_counts:
        print("\nTag distribution:")
        for tag, count in tag_counts.most_common():
            print(f"  {tag}: {count}")


if __name__ == "__main__":
    main()
