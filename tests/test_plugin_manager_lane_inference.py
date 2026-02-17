from pathlib import Path

from statistic_harness.core.plugin_manager import PluginManager


def _write_schema(path: Path) -> None:
    path.write_text('{"type":"object"}', encoding="utf-8")


def _write_manifest(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_lane_inference_defaults_to_decision_for_analysis(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.joinpath("plugin_manifest.schema.json").write_text(
        Path("docs/plugin_manifest.schema.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    plugin_dir = tmp_path / "plugins" / "demo_analysis"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    _write_schema(plugin_dir / "config.schema.json")
    _write_schema(plugin_dir / "output.schema.json")
    _write_manifest(
        plugin_dir / "plugin.yaml",
        """id: demo_analysis
name: Demo Analysis
version: 0.1.0
type: analysis
entrypoint: plugin.py:Plugin
depends_on: []
capabilities: []
config_schema: config.schema.json
output_schema: output.schema.json
sandbox:
  no_network: true
  fs_allowlist: ["appdata/"]
settings:
  description: demo
  defaults: {}
""",
    )
    manager = PluginManager(tmp_path / "plugins")
    specs = manager.discover()
    assert len(specs) == 1
    assert specs[0].lane == "decision"
    assert specs[0].decision_capable is True
    assert specs[0].requires_downstream_mapping is False


def test_lane_inference_diagnostic_only_analysis_is_explanation(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.joinpath("plugin_manifest.schema.json").write_text(
        Path("docs/plugin_manifest.schema.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    plugin_dir = tmp_path / "plugins" / "diag_analysis"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    _write_schema(plugin_dir / "config.schema.json")
    _write_schema(plugin_dir / "output.schema.json")
    _write_manifest(
        plugin_dir / "plugin.yaml",
        """id: diag_analysis
name: Diagnostic Analysis
version: 0.1.0
type: analysis
entrypoint: plugin.py:Plugin
depends_on: []
capabilities: ["diagnostic_only"]
config_schema: config.schema.json
output_schema: output.schema.json
sandbox:
  no_network: true
  fs_allowlist: ["appdata/"]
settings:
  description: demo
  defaults: {}
""",
    )
    manager = PluginManager(tmp_path / "plugins")
    specs = manager.discover()
    assert len(specs) == 1
    assert specs[0].lane == "explanation"
    assert specs[0].decision_capable is False
    assert specs[0].requires_downstream_mapping is True
