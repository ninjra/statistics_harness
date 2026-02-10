from __future__ import annotations

from pathlib import Path

import yaml

from scripts.top20_methods_matrix import _top20_plugin_ids


def test_top20_plugin_scaffold_exists_and_yaml_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    plugin_ids = _top20_plugin_ids()
    assert len(plugin_ids) == 20

    for pid in plugin_ids:
        pdir = root / "plugins" / pid
        assert (pdir / "plugin.yaml").exists()
        assert (pdir / "plugin.py").exists()
        assert (pdir / "config.schema.json").exists()
        assert (pdir / "output.schema.json").exists()

        manifest = yaml.safe_load((pdir / "plugin.yaml").read_text(encoding="utf-8"))
        assert isinstance(manifest, dict)
        assert manifest.get("id") == pid

