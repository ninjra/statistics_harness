from __future__ import annotations

from pathlib import Path

from statistic_harness.core.plugin_manager import PluginManager


def test_sqlpack_materialize_defaults_enabled_and_bootstrap_source() -> None:
    manager = PluginManager(Path("plugins"))
    specs = {spec.plugin_id: spec for spec in manager.discover()}
    spec = specs["transform_sqlpack_materialize_v1"]
    resolved = manager.resolve_config(spec, dict(spec.settings.get("defaults", {})))
    assert bool(resolved.get("enabled")) is True
    assert str(resolved.get("source_relpath") or "") == "artifacts/transform_sql_intents_pack_v1/sql_pack_bootstrap.json"
