from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


def _make_report(tmp_path: Path) -> Path:
    items = [
        {
            "action_type": "batch_input",
            "primary_process_id": "proc_a",
            "modeled_delta_hours": 10.0,
            "modeled_delta_hours_close_cycle": 5.0,
            "modeled_user_touches_reduced": 3.0,
            "modeled_contention_reduction_pct_close": 0.1,
            "value_score_v2": 0.8,
            "relevance_score": 0.7,
            "value_components": {
                "user_effort_score": 0.9,
                "close_window_score": 0.7,
                "server_contention_score": 0.5,
                "confidence_score": 0.6,
                "targeting_bonus": 0.3,
                "ambiguity_penalty": 0.05,
            },
        },
        {
            "action_type": "reschedule",
            "primary_process_id": "proc_b",
            "modeled_delta_hours": 5.0,
            "modeled_delta_hours_close_cycle": 2.0,
            "modeled_user_touches_reduced": 1.0,
            "modeled_contention_reduction_pct_close": 0.05,
            "value_score_v2": 0.6,
            "relevance_score": 0.5,
            "value_components": {
                "user_effort_score": 0.7,
                "close_window_score": 0.5,
                "server_contention_score": 0.3,
                "confidence_score": 0.4,
                "targeting_bonus": 0.2,
                "ambiguity_penalty": 0.03,
            },
        },
    ]
    report = {"recommendations": {"discovery": {"items": items}, "known": {"items": []}}}
    run_dir = tmp_path / "runs" / "test_run"
    run_dir.mkdir(parents=True)
    (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return run_dir


class TestCmdModel:
    def test_basic_model_json_output(self, tmp_path):
        run_dir = _make_report(tmp_path)
        output_path = tmp_path / "result.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "statistic_harness.cli",
                "model",
                str(run_dir),
                "--scenario",
                "test_scenario",
                "--format",
                "json",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["scenario_name"] == "test_scenario"

    def test_model_with_diff(self, tmp_path):
        run_dir = _make_report(tmp_path)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "statistic_harness.cli",
                "model",
                str(run_dir),
                "--scenario",
                "diff_test",
                "--close-cycles",
                "24",
                "--diff",
                "--format",
                "markdown",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Scenario Comparison" in result.stdout

    def test_model_with_top_n(self, tmp_path):
        run_dir = _make_report(tmp_path)
        output_path = tmp_path / "top1.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "statistic_harness.cli",
                "model",
                str(run_dir),
                "--scenario",
                "top1",
                "--top-n",
                "1",
                "--format",
                "json",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(output_path.read_text())
        assert data["items_after_filter"] == 1

    def test_batch_mode(self, tmp_path):
        run_dir = _make_report(tmp_path)
        scenarios_yaml = tmp_path / "scenarios.yaml"
        scenarios_yaml.write_text(
            yaml.dump(
                {
                    "scenarios": [
                        {"name": "s1", "close_cycles_per_year": 24},
                        {"name": "s2", "max_obviousness": 0.5},
                    ],
                    "compare": [{"baseline": "s1", "modeled": "s2"}],
                }
            )
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "statistic_harness.cli",
                "model",
                str(run_dir),
                "--batch",
                str(scenarios_yaml),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "s1" in result.stdout
        assert "s2" in result.stdout

    def test_model_with_weights(self, tmp_path):
        run_dir = _make_report(tmp_path)
        output_path = tmp_path / "weights.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "statistic_harness.cli",
                "model",
                str(run_dir),
                "--scenario",
                "custom_weights",
                "--weights",
                "value_score_v2=1.0",
                "--format",
                "json",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(output_path.read_text())
        assert data["scenario_name"] == "custom_weights"

    def test_model_suppress_actions(self, tmp_path):
        run_dir = _make_report(tmp_path)
        output_path = tmp_path / "suppressed.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "statistic_harness.cli",
                "model",
                str(run_dir),
                "--scenario",
                "no_reschedule",
                "--suppress-actions",
                "reschedule",
                "--format",
                "json",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(output_path.read_text())
        assert data["items_after_filter"] == 1
