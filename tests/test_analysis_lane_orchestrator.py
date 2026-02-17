import json
from pathlib import Path

from statistic_harness.core.pipeline import Pipeline
from statistic_harness.core.plugin_manager import PluginSpec


def _spec(
    tmp_path: Path,
    plugin_id: str,
    depends_on: list[str],
    lane: str,
) -> PluginSpec:
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
        lane=lane,
    )


def test_two_lane_plan_decision_first(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(tmp_path / "appdata"))
    pipeline = Pipeline(tmp_path / "appdata", Path("plugins"))
    d1 = _spec(tmp_path, "d1", [], "decision")
    d2 = _spec(tmp_path, "d2", ["d1"], "decision")
    e1 = _spec(tmp_path, "e1", ["d2"], "explanation")
    e2 = _spec(tmp_path, "e2", [], "explanation")
    plan = pipeline._plan_analysis_execution(
        [d1, d2, e1, e2],
        {"d1", "d2", "e1", "e2"},
        "two_lane_strict",
    )
    assert plan["mode_effective"] == "two_lane_strict"
    assert plan["fallback_reason"] == ""
    assert plan["decision_layers"] == [["d1"], ["d2"]]
    assert plan["explanation_layers"] == [["e1", "e2"]]


def test_two_lane_plan_falls_back_when_decision_depends_on_explanation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(tmp_path / "appdata"))
    pipeline = Pipeline(tmp_path / "appdata", Path("plugins"))
    d1 = _spec(tmp_path, "d1", ["e1"], "decision")
    e1 = _spec(tmp_path, "e1", [], "explanation")
    plan = pipeline._plan_analysis_execution(
        [d1, e1],
        {"d1", "e1"},
        "two_lane_strict",
    )
    assert plan["mode_effective"] == "legacy"
    assert plan["fallback_reason"] == "decision_depends_on_explanation"
    assert "d1->e1" in set(plan.get("fallback_edges") or [])
    assert plan["mixed_layers"] == [["e1"], ["d1"]]


def test_legacy_mode_uses_mixed_layers(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(tmp_path / "appdata"))
    pipeline = Pipeline(tmp_path / "appdata", Path("plugins"))
    d1 = _spec(tmp_path, "d1", [], "decision")
    e1 = _spec(tmp_path, "e1", ["d1"], "explanation")
    plan = pipeline._plan_analysis_execution([d1, e1], {"d1", "e1"}, "legacy")
    assert plan["mode_effective"] == "legacy"
    assert plan["fallback_reason"] == ""
    assert plan["mixed_layers"] == [["d1"], ["e1"]]
