from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "topo-tda-addon-pack-plan.md"
PLUGINS_DIR = REPO_ROOT / "plugins"


@dataclass(frozen=True)
class PluginCheck:
    plugin_id: str
    exists: bool
    has_plugin_yaml: bool
    has_plugin_py: bool
    has_config_schema: bool
    has_output_schema: bool


def _extract_analysis_plugin_ids(text: str) -> list[str]:
    # Only IDs that are explicitly named in the plan text.
    ids = sorted(set(re.findall(r"\banalysis_[a-z0-9_]+\b", text)))
    # The plan also includes wildcard directory globs like `plugins/analysis_tda_*/*`.
    # Filter out those prefix-only matches (e.g. "analysis_tda_").
    ids = [pid for pid in ids if not pid.endswith("_")]
    return ids


def _check_plugin(plugin_id: str) -> PluginCheck:
    base = PLUGINS_DIR / plugin_id
    exists = base.is_dir()
    return PluginCheck(
        plugin_id=plugin_id,
        exists=exists,
        has_plugin_yaml=(base / "plugin.yaml").exists() if exists else False,
        has_plugin_py=(base / "plugin.py").exists() if exists else False,
        has_config_schema=(base / "config.schema.json").exists() if exists else False,
        has_output_schema=(base / "output.schema.json").exists() if exists else False,
    )


def main() -> int:
    if not PLAN_PATH.exists():
        raise SystemExit(f"Missing plan file: {PLAN_PATH}")
    text = PLAN_PATH.read_text(encoding="utf-8")

    plugin_ids = _extract_analysis_plugin_ids(text)
    checks = [_check_plugin(pid) for pid in plugin_ids]
    missing_plugins = [c.plugin_id for c in checks if not c.exists]
    incomplete_plugins = [
        c.plugin_id
        for c in checks
        if c.exists
        and not (c.has_plugin_yaml and c.has_plugin_py and c.has_config_schema and c.has_output_schema)
    ]

    required_files = [
        # Plan “Location” items that aren’t plugins/* dirs.
        "src/statistic_harness/core/planner.py",
        "src/statistic_harness/core/stat_plugins/ideaspace.py",
        "tests/plugins/test_crosscutting_new_plugins.py",
        "src/statistic_harness/core/stat_plugins/surface.py",
    ]
    file_status = {p: (REPO_ROOT / p).exists() for p in required_files}

    out = {
        "plan_path": str(PLAN_PATH),
        "analysis_plugin_ids_in_plan": plugin_ids,
        "analysis_plugin_count_in_plan": len(plugin_ids),
        "missing_plugins": missing_plugins,
        "missing_plugins_count": len(missing_plugins),
        "incomplete_plugin_dirs": incomplete_plugins,
        "incomplete_plugin_dirs_count": len(incomplete_plugins),
        "required_files": file_status,
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
