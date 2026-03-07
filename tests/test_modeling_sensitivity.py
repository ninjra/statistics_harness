from __future__ import annotations

import json
from pathlib import Path

import pytest

from statistic_harness.core.modeling.engine import ModelingEngine
from statistic_harness.core.modeling.scenario import ScenarioConfig
from statistic_harness.core.modeling.sensitivity import render_sweep_table, sweep_variable


def _make_item(action_type: str, process_id: str, delta_hours: float = 5.0) -> dict:
    return {
        "action_type": action_type,
        "primary_process_id": process_id,
        "modeled_delta_hours": delta_hours,
        "modeled_delta_hours_close_cycle": delta_hours / 2,
        "modeled_user_touches_reduced": 2.0,
        "modeled_contention_reduction_pct_close": 0.1,
        "value_score_v2": 0.5,
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


def _build_engine(tmp_path: Path) -> ModelingEngine:
    items = [
        _make_item("batch_input", "p1", delta_hours=10.0),
        _make_item("reschedule", "p2", delta_hours=8.0),
        _make_item("tune_threshold", "p3", delta_hours=3.0),
        _make_item("route_process", "p4", delta_hours=6.0),
        _make_item("batch_group_candidate", "p5", delta_hours=7.0),
    ]
    report = {"recommendations": {"discovery": {"items": items}, "known": {"items": []}}}
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    engine = ModelingEngine(tmp_path)
    engine.load()
    return engine


class TestSweepVariable:
    def test_produces_correct_count(self, tmp_path):
        engine = _build_engine(tmp_path)
        values = [0.3, 0.5, 0.7, 0.9, 1.0]
        results = sweep_variable(engine, "max_obviousness", values)
        assert len(results) == 5
        for val, mr in results:
            assert val in values
            assert mr.scenario.max_obviousness == val

    def test_monotonic_filtering(self, tmp_path):
        engine = _build_engine(tmp_path)
        values = [0.2, 0.4, 0.6, 0.8, 1.0]
        results = sweep_variable(engine, "max_obviousness", values)
        counts = [len(mr.items) for _, mr in results]
        # More permissive threshold should yield >= items
        for i in range(len(counts) - 1):
            assert counts[i] <= counts[i + 1]

    def test_base_scenario_preserved(self, tmp_path):
        engine = _build_engine(tmp_path)
        base = ScenarioConfig(name="mybase", require_modeled_hours=True)
        results = sweep_variable(engine, "max_obviousness", [0.5, 1.0], base_scenario=base)
        for _, mr in results:
            assert mr.scenario.require_modeled_hours is True

    def test_top_n_sweep(self, tmp_path):
        engine = _build_engine(tmp_path)
        results = sweep_variable(engine, "discovery_top_n", [1, 2, 3, 5])
        for val, mr in results:
            assert len(mr.items) <= val


class TestRenderSweepTable:
    def test_renders_table(self, tmp_path):
        engine = _build_engine(tmp_path)
        results = sweep_variable(engine, "max_obviousness", [0.5, 1.0])
        table = render_sweep_table(results, top_k=3)
        assert "| Item |" in table
        assert "batch_input:p1" in table

    def test_empty_results(self):
        table = render_sweep_table([])
        assert "No sweep results" in table
