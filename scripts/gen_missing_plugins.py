from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "docs" / "codex_stat_plugins_spec_pack.md"
PLUGINS_DIR = ROOT / "plugins"


ALIAS_WRAPPERS = {
    "analysis_isolation_forest": "analysis_isolation_forest_anomaly",
    "analysis_log_template_drain": "analysis_log_template_mining_drain",
    "analysis_robust_pca_pcp": "analysis_robust_pca_sparse_outliers",
}


def extract_plugin_ids(text: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("##"):
            continue
        for token in re.findall(r"analysis_[a-z0-9_]+", line):
            if token not in seen:
                ids.append(token)
                seen.add(token)
    return ids


def title_from_id(plugin_id: str) -> str:
    label = plugin_id.replace("analysis_", "").replace("_", " ")
    return " ".join(word.capitalize() for word in label.split())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    spec_text = SPEC_PATH.read_text(encoding="utf-8")
    spec_ids = extract_plugin_ids(spec_text)
    existing = {p.name for p in PLUGINS_DIR.glob("*") if p.is_dir()}
    missing = [pid for pid in spec_ids if pid not in existing]

    created = []
    for plugin_id in missing:
        plugin_dir = PLUGINS_DIR / plugin_id
        ensure_dir(plugin_dir)
        (plugin_dir / "__init__.py").write_text("", encoding="utf-8")
        config_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": True,
            "properties": {},
        }
        output_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": True,
            "required": [
                "status",
                "summary",
                "metrics",
                "findings",
                "artifacts",
                "budget",
                "error",
                "references",
                "debug",
            ],
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
                "metrics": {"type": "object"},
                "findings": {"type": "array"},
                "artifacts": {"type": "array"},
                "budget": {"type": "object"},
                "error": {"type": ["object", "null"]},
                "references": {"type": "array"},
                "debug": {"type": "object"},
            },
        }
        write_json(plugin_dir / "config.schema.json", config_schema)
        write_json(plugin_dir / "output.schema.json", output_schema)
        plugin_yaml = (
            "id: {pid}\n"
            "name: {name}\n"
            "version: 0.1.0\n"
            "type: analysis\n"
            "entrypoint: plugin.py:Plugin\n"
            "depends_on: []\n"
            "settings:\n"
            "  description: Auto-generated plugin skeleton.\n"
            "  defaults: {{}}\n"
            "capabilities: []\n"
            "config_schema: config.schema.json\n"
            "output_schema: output.schema.json\n"
            "sandbox:\n"
            "  no_network: true\n"
            "  fs_allowlist:\n"
            "    - appdata\n"
            "    - plugins\n"
            "    - run_dir\n"
        ).format(pid=plugin_id, name=title_from_id(plugin_id))
        (plugin_dir / "plugin.yaml").write_text(plugin_yaml, encoding="utf-8")
        plugin_py = (
            "from __future__ import annotations\n\n"
            "from statistic_harness.core.stat_plugins.registry import run_plugin\n\n\n"
            "class Plugin:\n"
            "    def run(self, ctx):\n"
            "        return run_plugin(\"{pid}\", ctx)\n"
        ).format(pid=plugin_id)
        (plugin_dir / "plugin.py").write_text(plugin_py, encoding="utf-8")
        created.append(plugin_id)

    print(f"created_count={len(created)}")
    if created:
        print("created:")
        for pid in created:
            print(f"- {pid}")

    for alias, base in ALIAS_WRAPPERS.items():
        if alias in existing or alias in created:
            continue
        plugin_dir = PLUGINS_DIR / alias
        ensure_dir(plugin_dir)
        (plugin_dir / "__init__.py").write_text("", encoding="utf-8")
        write_json(plugin_dir / "config.schema.json", config_schema)
        write_json(plugin_dir / "output.schema.json", output_schema)
        plugin_yaml = (
            "id: {pid}\n"
            "name: {name}\n"
            "version: 0.1.0\n"
            "type: analysis\n"
            "entrypoint: plugin.py:Plugin\n"
            "depends_on: []\n"
            "settings:\n"
            "  description: Alias wrapper for {base}.\n"
            "  defaults: {{}}\n"
            "capabilities: []\n"
            "config_schema: config.schema.json\n"
            "output_schema: output.schema.json\n"
            "sandbox:\n"
            "  no_network: true\n"
            "  fs_allowlist:\n"
            "    - appdata\n"
            "    - plugins\n"
            "    - run_dir\n"
        ).format(pid=alias, name=title_from_id(alias), base=base)
        (plugin_dir / "plugin.yaml").write_text(plugin_yaml, encoding="utf-8")
        plugin_py = (
            "from __future__ import annotations\n\n"
            "from statistic_harness.core.stat_plugins.registry import run_plugin\n\n\n"
            "class Plugin:\n"
            "    def run(self, ctx):\n"
            "        return run_plugin(\"{pid}\", ctx)\n"
        ).format(pid=alias)
        (plugin_dir / "plugin.py").write_text(plugin_py, encoding="utf-8")
        created.append(alias)


if __name__ == "__main__":
    main()
