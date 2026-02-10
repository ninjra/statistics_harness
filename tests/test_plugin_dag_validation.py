import json
from pathlib import Path

import pytest

from statistic_harness.core.pipeline import Pipeline
from statistic_harness.core.plugin_manager import PluginSpec


def _spec(tmp_path: Path, plugin_id: str, depends_on: list[str]) -> PluginSpec:
    schema = tmp_path / f"{plugin_id}.schema.json"
    schema.write_text(json.dumps({"type": "object"}), encoding="utf-8")
    out = tmp_path / f"{plugin_id}.out.json"
    out.write_text(json.dumps({"type": "object"}), encoding="utf-8")
    return PluginSpec(
        plugin_id=plugin_id,
        name=plugin_id,
        version="0.0.0",
        type="analysis",
        entrypoint="plugin:Plugin",
        depends_on=depends_on,
        settings={},
        path=tmp_path,
        capabilities=[],
        config_schema=schema,
        output_schema=out,
        sandbox={},
    )


def test_toposort_cycle_error_includes_path_and_edges(tmp_path, monkeypatch):
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(tmp_path / "appdata"))
    pipeline = Pipeline(tmp_path / "appdata", Path("plugins"))
    a = _spec(tmp_path, "A", ["B"])
    b = _spec(tmp_path, "B", ["A"])
    with pytest.raises(ValueError) as excinfo:
        pipeline._toposort_layers([a, b], {"A", "B"})
    msg = str(excinfo.value)
    assert "Cycle detected" in msg
    assert "cycle=" in msg
    assert "edges=" in msg


def test_expand_selected_with_deps_reports_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(tmp_path / "appdata"))
    pipeline = Pipeline(tmp_path / "appdata", Path("plugins"))
    a = _spec(tmp_path, "A", ["B"])
    expanded, added, missing = pipeline._expand_selected_with_deps({"A": a}, {"A"})
    assert expanded == {"A"}
    assert added == []
    assert missing == ["A -> B"]

