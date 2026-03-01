#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _discover_plugin_ids(plugins_dir: Path) -> list[str]:
    return sorted(path.parent.name for path in plugins_dir.glob("*/plugin.yaml"))


def _load_exemptions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    if isinstance(payload, list):
        return {str(item).strip() for item in payload if str(item).strip()}
    if not isinstance(payload, dict):
        return set()
    exempt = payload.get("exempt_plugin_ids")
    if not isinstance(exempt, list):
        return set()
    return {str(item).strip() for item in exempt if str(item).strip()}


def _load_test_text(tests_root: Path) -> str:
    parts: list[str] = []
    for path in sorted(tests_root.rglob("*.py")):
        try:
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
    return "\n".join(parts)


def verify_plugin_test_coverage(
    *,
    plugins_dir: Path,
    tests_root: Path,
    exemptions_path: Path,
) -> dict[str, Any]:
    plugin_ids = _discover_plugin_ids(plugins_dir)
    plugin_set = set(plugin_ids)
    exemptions = sorted(_load_exemptions(exemptions_path))
    exemptions_set = set(exemptions)
    stale_exemptions = sorted(exemptions_set - plugin_set)

    test_text = _load_test_text(tests_root)
    covered: list[str] = []
    for plugin_id in plugin_ids:
        if re.search(r"\b" + re.escape(plugin_id) + r"\b", test_text):
            covered.append(plugin_id)

    covered_set = set(covered)
    uncovered_all = sorted(plugin_set - covered_set)
    uncovered_unexempted = sorted(pid for pid in uncovered_all if pid not in exemptions_set)
    uncovered_exempted = sorted(pid for pid in uncovered_all if pid in exemptions_set)

    plugin_count = len(plugin_ids)
    covered_count = len(covered_set)
    coverage_ratio = (float(covered_count) / float(plugin_count)) if plugin_count else 1.0

    return {
        "schema_version": "plugin_test_coverage.v1",
        "ok": len(uncovered_unexempted) == 0,
        "plugins_dir": str(plugins_dir),
        "tests_root": str(tests_root),
        "exemptions_path": str(exemptions_path),
        "plugin_count": plugin_count,
        "covered_count": covered_count,
        "coverage_ratio": round(coverage_ratio, 6),
        "covered_plugins": sorted(covered_set),
        "uncovered_count": len(uncovered_all),
        "uncovered_plugins": uncovered_all,
        "uncovered_unexempted_count": len(uncovered_unexempted),
        "uncovered_unexempted_plugins": uncovered_unexempted,
        "uncovered_exempted_count": len(uncovered_exempted),
        "uncovered_exempted_plugins": uncovered_exempted,
        "exemptions_count": len(exemptions_set),
        "exemptions": exemptions,
        "stale_exemptions_count": len(stale_exemptions),
        "stale_exemptions": stale_exemptions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify plugin-to-test coverage with optional explicit exemptions."
    )
    parser.add_argument("--plugins-dir", default=str(ROOT / "plugins"))
    parser.add_argument("--tests-root", default=str(ROOT / "tests"))
    parser.add_argument(
        "--exemptions-json",
        default=str(ROOT / "config" / "plugin_test_coverage_exemptions.json"),
    )
    parser.add_argument("--out-json", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    plugins_dir = Path(str(args.plugins_dir))
    tests_root = Path(str(args.tests_root))
    exemptions_path = Path(str(args.exemptions_json))

    payload = verify_plugin_test_coverage(
        plugins_dir=plugins_dir,
        tests_root=tests_root,
        exemptions_path=exemptions_path,
    )

    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out_json = str(args.out_json).strip()
    if out_json:
        out_path = Path(out_json)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")

    print(rendered, end="")
    if bool(args.strict) and not bool(payload.get("ok")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
