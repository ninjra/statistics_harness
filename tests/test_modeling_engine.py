from __future__ import annotations

import json
from pathlib import Path

import pytest

from statistic_harness.core.modeling.engine import ModelingEngine
from statistic_harness.core.modeling.scenario import ScenarioConfig


def _make_item(
    action_type: str = "batch_input",
    process_id: str = "proc_a",
    modeled_delta_hours: float = 10.0,
    modeled_delta_hours_close_cycle: float = 5.0,
    modeled_user_touches_reduced: float = 3.0,
    modeled_contention_reduction_pct_close: float = 0.1,
    value_score_v2: float = 0.8,
    relevance_score: float = 0.7,
    value_components: dict | None = None,
) -> dict:
    return {
        "action_type": action_type,
        "primary_process_id": process_id,
        "modeled_delta_hours": modeled_delta_hours,
        "modeled_delta_hours_close_cycle": modeled_delta_hours_close_cycle,
        "modeled_user_touches_reduced": modeled_user_touches_reduced,
        "modeled_contention_reduction_pct_close": modeled_contention_reduction_pct_close,
        "value_score_v2": value_score_v2,
        "relevance_score": relevance_score,
        "value_components": value_components
        or {
            "user_effort_score": 0.9,
            "close_window_score": 0.7,
            "server_contention_score": 0.5,
            "confidence_score": 0.6,
            "targeting_bonus": 0.3,
            "ambiguity_penalty": 0.05,
        },
    }


def _write_report(tmp_path: Path, items: list[dict]) -> Path:
    report = {
        "recommendations": {
            "discovery": {"items": items},
            "known": {"items": []},
        }
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return tmp_path


@pytest.fixture()
def engine_with_items(tmp_path):
    items = [
        _make_item("batch_input", "proc_a", modeled_delta_hours=10.0, modeled_delta_hours_close_cycle=5.0, value_score_v2=0.8),
        _make_item("reschedule", "proc_b", modeled_delta_hours=8.0, modeled_delta_hours_close_cycle=4.0, value_score_v2=0.6),
        _make_item("tune_threshold", "proc_c", modeled_delta_hours=2.0, modeled_delta_hours_close_cycle=1.0, value_score_v2=0.3),
        _make_item("route_process", "proc_d", modeled_delta_hours=0.0, modeled_delta_hours_close_cycle=3.0, value_score_v2=0.5),
        _make_item("batch_group_candidate", "proc_e", modeled_delta_hours=6.0, modeled_delta_hours_close_cycle=2.0, value_score_v2=0.9),
    ]
    run_dir = _write_report(tmp_path, items)
    engine = ModelingEngine(run_dir)
    engine.load()
    return engine


class TestModelingEngineLoad:
    def test_load_missing_report(self, tmp_path):
        engine = ModelingEngine(tmp_path)
        with pytest.raises(FileNotFoundError):
            engine.load()

    def test_load_extracts_items(self, engine_with_items):
        assert engine_with_items._raw_items is not None
        assert len(engine_with_items._raw_items) == 5

    def test_items_tagged_with_source_lane(self, engine_with_items):
        for item in engine_with_items._raw_items:
            assert item["_source_lane"] == "discovery"

    def test_load_known_items(self, tmp_path):
        report = {
            "recommendations": {
                "discovery": {"items": [_make_item("batch_input", "d1")]},
                "known": {"items": [_make_item("reschedule", "k1")]},
            }
        }
        (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
        engine = ModelingEngine(tmp_path)
        engine.load()
        lanes = {item["_source_lane"] for item in engine._raw_items}
        assert lanes == {"discovery", "known"}


class TestRescaleCloseCycle:
    def test_no_rescale_when_none(self, engine_with_items):
        config = ScenarioConfig(name="baseline")
        result = engine_with_items.run_scenario(config)
        for item in result.items:
            assert "_close_cycle_rescaled" not in item

    def test_rescale_doubles_at_24_cycles(self, engine_with_items):
        config = ScenarioConfig(name="double", close_cycles_per_year=24.0)
        result = engine_with_items.run_scenario(config)
        rescaled = [i for i in result.items if i.get("_close_cycle_rescaled")]
        assert len(rescaled) > 0
        # Original proc_a had close_cycle=5.0, at 24 cycles -> 10.0
        proc_a = [i for i in result.items if i["primary_process_id"] == "proc_a"][0]
        assert proc_a["modeled_delta_hours_close_cycle"] == pytest.approx(10.0)

    def test_rescale_halves_at_6_cycles(self, engine_with_items):
        config = ScenarioConfig(name="half", close_cycles_per_year=6.0)
        result = engine_with_items.run_scenario(config)
        proc_a = [i for i in result.items if i["primary_process_id"] == "proc_a"][0]
        assert proc_a["modeled_delta_hours_close_cycle"] == pytest.approx(2.5)


class TestRecomputeValueScores:
    def test_no_recompute_when_none(self, engine_with_items):
        config = ScenarioConfig(name="baseline")
        result = engine_with_items.run_scenario(config)
        for item in result.items:
            assert "_value_score_recomputed" not in item

    def test_recompute_with_custom_weights(self, engine_with_items):
        config = ScenarioConfig(
            name="reweight",
            value_component_weights={
                "user_effort_score": 1.0,
                "close_window_score": 0.0,
                "server_contention_score": 0.0,
                "confidence_score": 0.0,
                "targeting_bonus": 0.0,
            },
        )
        result = engine_with_items.run_scenario(config)
        recomputed = [i for i in result.items if i.get("_value_score_recomputed")]
        assert len(recomputed) > 0
        # All items have user_effort_score=0.9, penalty=0.05 -> value = 1.0*0.9 - 0.05 = 0.85
        for item in recomputed:
            assert item["value_score_v2"] == pytest.approx(0.85)


class TestRecomputeWeightedRanks:
    def test_default_weights_applied(self, engine_with_items):
        config = ScenarioConfig(name="baseline")
        result = engine_with_items.run_scenario(config)
        for item in result.items:
            assert "weighted_rank_score" in item
            assert item["_weighted_rank_recomputed"] is True

    def test_custom_weights(self, engine_with_items):
        config = ScenarioConfig(
            name="value_only",
            ranking_weights={"value_score_v2": 1.0},
        )
        result = engine_with_items.run_scenario(config)
        # proc_e has highest value_score_v2=0.9
        assert result.items[0]["primary_process_id"] == "proc_e"
        assert result.items[0]["weighted_rank_score"] == pytest.approx(0.9)


class TestApplyFilters:
    def test_suppress_action_types(self, engine_with_items):
        config = ScenarioConfig(
            name="suppress",
            suppress_action_types=frozenset({"tune_threshold"}),
        )
        result = engine_with_items.run_scenario(config)
        action_types = {i["action_type"] for i in result.items}
        assert "tune_threshold" not in action_types

    def test_max_obviousness(self, engine_with_items):
        config = ScenarioConfig(name="low_obv", max_obviousness=0.5)
        result = engine_with_items.run_scenario(config)
        # batch_input (0.30) and batch_group_candidate (0.20) should survive
        # route_process (0.60), reschedule (0.70), tune_threshold (0.95) filtered
        action_types = {i["action_type"] for i in result.items}
        assert "batch_input" in action_types
        assert "batch_group_candidate" in action_types
        assert "tune_threshold" not in action_types
        assert "reschedule" not in action_types

    def test_require_modeled_hours(self, engine_with_items):
        config = ScenarioConfig(name="hours", require_modeled_hours=True)
        result = engine_with_items.run_scenario(config)
        # proc_d has modeled_delta_hours=0.0 -> filtered
        pids = {i["primary_process_id"] for i in result.items}
        assert "proc_d" not in pids

    def test_top_n(self, engine_with_items):
        config = ScenarioConfig(name="top2", discovery_top_n=2)
        result = engine_with_items.run_scenario(config)
        assert len(result.items) == 2

    def test_allow_action_types(self, engine_with_items):
        config = ScenarioConfig(
            name="allow_only",
            allow_action_types=frozenset({"batch_input", "reschedule"}),
        )
        result = engine_with_items.run_scenario(config)
        for item in result.items:
            assert item["action_type"] in {"batch_input", "reschedule"}

    def test_per_action_cap(self, tmp_path):
        items = [
            _make_item("batch_input", f"proc_{i}", modeled_delta_hours=float(10 - i))
            for i in range(5)
        ]
        run_dir = _write_report(tmp_path, items)
        engine = ModelingEngine(run_dir)
        engine.load()
        config = ScenarioConfig(name="capped", max_per_action_type={"batch_input": 2})
        result = engine.run_scenario(config)
        assert len(result.items) == 2


class TestSortOrder:
    def test_sorted_by_weighted_rank_descending(self, engine_with_items):
        config = ScenarioConfig(name="baseline")
        result = engine_with_items.run_scenario(config)
        scores = [i["weighted_rank_score"] for i in result.items]
        assert scores == sorted(scores, reverse=True)


class TestScenarioIsolation:
    def test_scenarios_dont_mutate_each_other(self, engine_with_items):
        r1 = engine_with_items.run_scenario(
            ScenarioConfig(name="s1", close_cycles_per_year=24.0)
        )
        r2 = engine_with_items.run_scenario(
            ScenarioConfig(name="s2", close_cycles_per_year=6.0)
        )
        proc_a_r1 = [i for i in r1.items if i["primary_process_id"] == "proc_a"][0]
        proc_a_r2 = [i for i in r2.items if i["primary_process_id"] == "proc_a"][0]
        assert proc_a_r1["modeled_delta_hours_close_cycle"] == pytest.approx(10.0)
        assert proc_a_r2["modeled_delta_hours_close_cycle"] == pytest.approx(2.5)
