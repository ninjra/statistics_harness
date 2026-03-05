"""Evaluator harness integration tests (four-pillars Task 4.2).

Validates the evaluation.py module that checks reports against ground truth
YAML files, including tolerance matching, finding validation, and edge cases.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from statistic_harness.core.evaluation import (
    _matches_expected,
    _parse_tolerance,
    _within_tolerance,
    evaluate_report,
)


def _write_report(tmp_path: Path, report: dict) -> Path:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _write_ground_truth(tmp_path: Path, truth: dict) -> Path:
    path = tmp_path / "ground_truth.yaml"
    path.write_text(yaml.dump(truth), encoding="utf-8")
    return path


class TestParseAndWithinTolerance:
    """Unit tests for tolerance parsing and matching."""

    def test_parse_none_returns_defaults(self) -> None:
        tol = _parse_tolerance(None)
        assert tol == {"absolute": 0.0, "relative": 0.0}

    def test_parse_numeric_as_absolute(self) -> None:
        tol = _parse_tolerance(5.0)
        assert tol["absolute"] == 5.0
        assert tol["relative"] == 0.0

    def test_parse_dict_absolute_and_relative(self) -> None:
        tol = _parse_tolerance({"absolute": 2.0, "relative": 0.1})
        assert tol["absolute"] == 2.0
        assert tol["relative"] == 0.1

    def test_within_tolerance_exact_match(self) -> None:
        tol = {"absolute": 0.0, "relative": 0.0}
        assert _within_tolerance(10.0, 10.0, tol) is True

    def test_within_tolerance_absolute(self) -> None:
        tol = {"absolute": 1.0, "relative": 0.0}
        assert _within_tolerance(10.0, 10.5, tol) is True
        assert _within_tolerance(10.0, 12.0, tol) is False

    def test_within_tolerance_relative(self) -> None:
        tol = {"absolute": 0.0, "relative": 0.1}
        assert _within_tolerance(100.0, 105.0, tol) is True
        assert _within_tolerance(100.0, 115.0, tol) is False

    def test_within_tolerance_absolute_takes_priority(self) -> None:
        tol = {"absolute": 5.0, "relative": 0.0}
        assert _within_tolerance(100.0, 104.0, tol) is True


class TestMatchesExpected:
    """Unit tests for finding pattern matching."""

    def test_where_exact_match(self) -> None:
        item = {"kind": "feature_discovery", "feature": "x1"}
        assert _matches_expected(item, {"feature": "x1"}, None) is True

    def test_where_mismatch(self) -> None:
        item = {"kind": "feature_discovery", "feature": "x1"}
        assert _matches_expected(item, {"feature": "x2"}, None) is False

    def test_contains_substring(self) -> None:
        item = {"kind": "anomaly", "description": "outlier at row 10"}
        assert _matches_expected(item, None, {"description": "outlier"}) is True

    def test_contains_list_membership(self) -> None:
        item = {"kind": "cluster", "members": ["a", "b", "c"]}
        assert _matches_expected(item, None, {"members": "b"}) is True

    def test_contains_subset(self) -> None:
        item = {"kind": "cluster", "members": ["a", "b", "c"]}
        assert _matches_expected(item, None, {"members": ["a", "c"]}) is True


@pytest.mark.slow
class TestEvaluateReport:
    """Integration tests for the full evaluate_report function."""

    def test_empty_ground_truth_produces_no_failures(self, tmp_path: Path) -> None:
        """An empty ground truth (no expectations) should pass."""
        report = {"plugins": {}, "recommendations": {"items": []}}
        report_path = _write_report(tmp_path, report)
        truth_path = _write_ground_truth(tmp_path, {"strict": False})
        ok, messages = evaluate_report(report_path, truth_path)
        assert ok is True
        assert len(messages) == 0

    def test_known_finding_matches_ground_truth(self, tmp_path: Path) -> None:
        """A report with the expected feature finding should pass."""
        report = {
            "plugins": {
                "analysis_test": {
                    "findings": [
                        {"kind": "feature_discovery", "feature": "x1", "id": "f1", "severity": "info"},
                    ]
                }
            }
        }
        truth = {
            "strict": False,
            "features": ["x1"],
        }
        report_path = _write_report(tmp_path, report)
        truth_path = _write_ground_truth(tmp_path, truth)
        ok, messages = evaluate_report(report_path, truth_path)
        assert ok is True, f"Unexpected failures: {messages}"

    def test_missing_finding_fails_ground_truth(self, tmp_path: Path) -> None:
        """When an expected feature is absent, evaluation should fail."""
        report = {
            "plugins": {
                "analysis_test": {
                    "findings": [
                        {"kind": "feature_discovery", "feature": "x1", "id": "f1", "severity": "info"},
                    ]
                }
            }
        }
        truth = {
            "strict": False,
            "features": ["x1", "x2"],
        }
        report_path = _write_report(tmp_path, report)
        truth_path = _write_ground_truth(tmp_path, truth)
        ok, messages = evaluate_report(report_path, truth_path)
        assert ok is False
        assert any("x2" in m for m in messages)

    def test_tolerance_matching_for_changepoints(self, tmp_path: Path) -> None:
        """Changepoint detection should use tolerance-based matching."""
        report = {
            "plugins": {
                "cp_plugin": {
                    "findings": [
                        {"kind": "changepoint", "index": 102},
                    ]
                }
            }
        }
        truth = {
            "strict": False,
            "changepoints": [100],
            "changepoint_tolerance": 5,
        }
        report_path = _write_report(tmp_path, report)
        truth_path = _write_ground_truth(tmp_path, truth)
        ok, messages = evaluate_report(report_path, truth_path)
        assert ok is True, f"Tolerance matching failed: {messages}"

    def test_strict_mode_flags_unexpected_features(self, tmp_path: Path) -> None:
        """In strict mode, unexpected features should be flagged."""
        report = {
            "plugins": {
                "analysis_test": {
                    "findings": [
                        {"kind": "feature_discovery", "feature": "x1", "id": "f1", "severity": "info"},
                        {"kind": "feature_discovery", "feature": "surprise", "id": "f2", "severity": "info"},
                    ]
                }
            }
        }
        truth = {
            "strict": True,
            "features": ["x1"],
        }
        report_path = _write_report(tmp_path, report)
        truth_path = _write_ground_truth(tmp_path, truth)
        ok, messages = evaluate_report(report_path, truth_path)
        assert ok is False
        assert any("surprise" in m.lower() or "Unexpected" in m for m in messages)

    def test_expected_metric_validation(self, tmp_path: Path) -> None:
        """Expected metrics must match within tolerance."""
        report = {
            "plugins": {
                "plugin_a": {
                    "findings": [],
                    "metrics": {"accuracy": 0.95},
                }
            }
        }
        truth = {
            "strict": False,
            "expected_metrics": [
                {"plugin_id": "plugin_a", "metric": "accuracy", "value": 0.94, "tolerance": 0.02},
            ],
        }
        report_path = _write_report(tmp_path, report)
        truth_path = _write_ground_truth(tmp_path, truth)
        ok, messages = evaluate_report(report_path, truth_path)
        assert ok is True, f"Metric validation failed: {messages}"
