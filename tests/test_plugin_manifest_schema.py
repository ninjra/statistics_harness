from pathlib import Path

import pytest
from jsonschema import ValidationError

from statistic_harness.core.plugin_manager import PluginManager


def test_manifest_schema_validation(tmp_path: Path) -> None:
    schema_src = Path("docs/plugin_manifest.schema.json")
    schema_dst = tmp_path / "docs" / "plugin_manifest.schema.json"
    schema_dst.parent.mkdir(parents=True, exist_ok=True)
    schema_dst.write_text(schema_src.read_text(encoding="utf-8"), encoding="utf-8")

    plugins_dir = tmp_path / "plugins"
    plugin_dir = plugins_dir / "bad_plugin"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "config.schema.json").write_text("{}", encoding="utf-8")
    (plugin_dir / "output.schema.json").write_text("{}", encoding="utf-8")
    (plugin_dir / "plugin.yaml").write_text(
        """id: bad_plugin
name: Bad Plugin
version: 0.1.0
type: analysis
entrypoint: plugin.py:Plugin
""",
        encoding="utf-8",
    )

    manager = PluginManager(plugins_dir)
    with pytest.raises(ValueError, match="Invalid manifest"):
        manager.discover()


def test_plugin_config_and_output_validation() -> None:
    manager = PluginManager(Path("plugins"))
    specs = {spec.plugin_id: spec for spec in manager.discover()}
    ingest = specs["ingest_tabular"]

    manager.validate_config(
        ingest,
        {"encoding": "utf-8", "delimiter": None, "sheet_name": None, "chunk_size": 10},
    )
    with pytest.raises(ValidationError):
        manager.validate_config(ingest, {"chunk_size": "bad"})

    manager.validate_output(
        ingest,
        {
            "status": "ok",
            "summary": "ok",
            "metrics": {},
            "findings": [],
            "artifacts": [],
            "budget": {
                "row_limit": None,
                "sampled": False,
                "time_limit_ms": None,
                "cpu_limit_ms": None,
            },
            "error": None,
        },
    )
