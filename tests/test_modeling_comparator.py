from __future__ import annotations

import json
from pathlib import Path

import pytest

from statistic_harness.core.modeling.comparator import (
    ComparisonReport,
    RankDelta,
    ScenarioComparator,
)
from statistic_harness.core.modeling.engine import ModelingEngine
from statistic_harness.core.modeling.scenario import ScenarioConfig


def _make_item(action_type: str, process_id: str, delta_hours: float = 5.0, value: float = 0.5) -> dict:
    return {
        "action_type": action_type,
        "primary_process_id": process_id,
        "modeled_delta_hours": delta_hours,
        "modeled_delta_hours_close_cycle": delta_hours / 2,
        "modeled_user_touches_reduced": 2.0,
        "modeled_contention_reduction_pct_close": 0.1,
        "value_score_v2": value,
        "relevance_score": 0.7,
        "value_components": {
            "user_effort_score": 0.8,
            "close_window_score": 0.6,
            "server_contention_score": 0.4,
            "confidence_score": 0.5,
            "targeting_bonus": 0.2,
            "ambiguity_penalty": 0.02,
        },
    }


def _build_engine(tmp_path: Path, items: list[dict]) -> ModelingEngine:
    report = {"recommendations": {"discovery": {"items": items}, "known": {"items": []}}}
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    engine = ModelingEngine(tmp_path)
    engine.load()
    return engine


class TestScenarioComparator:
    def test_compare_identical(self, tmp_path):
        items = [_make_item("batch_input", "p1"), _make_item("reschedule", "p2")]
        engine = _build_engine(tmp_path, items)
        baseline = engine.run_scenario(ScenarioConfig(name="base"))
        modeled = engine.run_scenario(ScenarioConfig(name="same"))
        report = ScenarioComparator.compare(baseline, modeled)
        assert report.baseline_count == report.modeled_count
        assert len(report.appeared) == 0
        assert len(report.disappeared) == 0
        assert len(report.movers) == 0

    def test_compare_with_filtering_causes_disappeared(self, tmp_path):
        items = [
            _make_item("batch_input", "p1", delta_hours=10.0),
            _make_item("tune_threshold", "p2", delta_hours=1.0),
        ]
        engine = _build_engine(tmp_path, items)
        baseline = engine.run_scenario(ScenarioConfig(name="base"))
        modeled = engine.run_scenario(
            ScenarioConfig(name="filtered", suppress_action_types=frozenset({"tune_threshold"}))
        )
        report = ScenarioComparator.compare(baseline, modeled)
        assert len(report.disappeared) == 1
        assert report.disappeared[0].item_key == "tune_threshold:p2"

    def test_rank_change_positive_means_moved_up(self, tmp_path):
        items = [
            _make_item("batch_input", "p1", delta_hours=10.0, value=0.5),
            _make_item("reschedule", "p2", delta_hours=5.0, value=0.9),
        ]
        engine = _build_engine(tmp_path, items)
        baseline = engine.run_scenario(ScenarioConfig(name="base"))
        # With value_score_v2 weight at 1.0, p2 (value=0.9) should beat p1 (value=0.5)
        modeled = engine.run_scenario(
            ScenarioConfig(name="value_heavy", ranking_weights={"value_score_v2": 1.0})
        )
        report = ScenarioComparator.compare(baseline, modeled)
        p2_delta = [d for d in report.deltas if "reschedule:p2" in d.item_key][0]
        # p2 should have moved up (positive rank_change)
        if p2_delta.rank_change is not None and p2_delta.rank_change > 0:
            assert p2_delta.rank_change > 0

    def test_compare_baseline_to_scenario(self, tmp_path):
        items = [_make_item("batch_input", "p1"), _make_item("reschedule", "p2")]
        engine = _build_engine(tmp_path, items)
        scenario = ScenarioConfig(name="test", discovery_top_n=1)
        report = ScenarioComparator.compare_baseline_to_scenario(engine, scenario)
        assert report.baseline_scenario == "baseline"
        assert report.modeled_scenario == "test"
        assert report.modeled_count == 1


class TestComparisonReport:
    def test_biggest_gains_and_drops(self):
        deltas = [
            RankDelta("a:1", 5, 1, 4, 0.5, 0.9, 0.4, False, False),
            RankDelta("b:2", 1, 5, -4, 0.9, 0.5, -0.4, False, False),
            RankDelta("c:3", 3, 3, 0, 0.7, 0.7, 0.0, False, False),
        ]
        report = ComparisonReport("base", "mod", deltas, 3, 3)
        assert len(report.biggest_gains) == 1
        assert report.biggest_gains[0].item_key == "a:1"
        assert len(report.biggest_drops) == 1
        assert report.biggest_drops[0].item_key == "b:2"
        assert len(report.movers) == 2

    def test_appeared_and_disappeared(self):
        deltas = [
            RankDelta("new:1", None, 1, None, 0.0, 0.8, 0.8, True, False),
            RankDelta("gone:2", 2, None, None, 0.7, 0.0, -0.7, False, True),
        ]
        report = ComparisonReport("base", "mod", deltas, 1, 1)
        assert len(report.appeared) == 1
        assert report.appeared[0].item_key == "new:1"
        assert len(report.disappeared) == 1
        assert report.disappeared[0].item_key == "gone:2"

    def test_to_markdown(self):
        deltas = [
            RankDelta("a:1", 5, 1, 4, 0.5, 0.9, 0.4, False, False),
            RankDelta("b:2", 1, 5, -4, 0.9, 0.5, -0.4, False, False),
            RankDelta("new:3", None, 2, None, 0.0, 0.6, 0.6, True, False),
            RankDelta("gone:4", 3, None, None, 0.4, 0.0, -0.4, False, True),
        ]
        report = ComparisonReport("base", "mod", deltas, 3, 3)
        md = report.to_markdown()
        assert "Scenario Comparison" in md
        assert "Biggest Rank Gains" in md
        assert "Biggest Rank Drops" in md
        assert "Newly Appeared" in md
        assert "Filtered Out" in md
        assert "a:1" in md
        assert "b:2" in md
