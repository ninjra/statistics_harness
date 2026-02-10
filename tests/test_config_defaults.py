import json
from pathlib import Path

from statistic_harness.core.plugin_manager import PluginManager, PluginSpec


def test_resolve_config_applies_jsonschema_defaults(tmp_path):
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    manager = PluginManager(plugins_dir)

    config_schema = tmp_path / "config.schema.json"
    config_schema.write_text(
        json.dumps(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "threshold": {"type": "number", "default": 0.5},
                    "nested": {
                        "type": "object",
                        "default": {},
                        "additionalProperties": False,
                        "properties": {
                            "mode": {"type": "string", "default": "auto"},
                        },
                    },
                },
                "required": ["nested"],
            }
        ),
        encoding="utf-8",
    )

    output_schema = tmp_path / "output.schema.json"
    output_schema.write_text(json.dumps({"type": "object"}), encoding="utf-8")

    spec = PluginSpec(
        plugin_id="test_plugin",
        name="Test Plugin",
        version="0.0.0",
        type="analysis",
        entrypoint="plugin:Plugin",
        depends_on=[],
        settings={},
        path=Path("."),
        capabilities=[],
        config_schema=config_schema,
        output_schema=output_schema,
        sandbox={},
    )

    resolved = manager.resolve_config(spec, {"nested": {}})
    assert resolved["threshold"] == 0.5
    assert resolved["nested"]["mode"] == "auto"

