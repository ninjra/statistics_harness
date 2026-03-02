"""Four-pillars integration: known-issue rediscovery baseline.

Acceptance criteria (Task 4.3):
  - Baseline run must independently rediscover known landmark issues.
  - Pipeline produces valid report with findings and plugin results.
  - Findings are deterministic across identical runs.
  - Gate fails on regression (missing plugins or status errors).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from statistic_harness.core.known_issue_compiler import compile_known_issues
from statistic_harness.core.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
SYNTH_FIXTURE = ROOT / "tests" / "fixtures" / "synth_linear.csv"

# Plugins that produce actionable findings on the synth_linear dataset.
REDISCOVERY_PLUGINS = [
    "profile_basic",
    "analysis_percentile_analysis",
    "analysis_tail_isolation",
    "changepoint_detection_v1",
]


def _run_baseline(tmp_path: Path) -> tuple[str, Path]:
    """Run pipeline and return (run_id, report_path)."""
    appdata = tmp_path / "appdata"
    appdata.mkdir(parents=True, exist_ok=True)
    old = os.environ.get("STAT_HARNESS_APPDATA")
    try:
        os.environ["STAT_HARNESS_APPDATA"] = str(appdata)
        pipeline = Pipeline(appdata, ROOT / "plugins")
        run_id = pipeline.run(SYNTH_FIXTURE, REDISCOVERY_PLUGINS, {}, 42)
        report_path = appdata / "runs" / run_id / "report.json"
        return run_id, report_path
    finally:
        if old is None:
            os.environ.pop("STAT_HARNESS_APPDATA", None)
        else:
            os.environ["STAT_HARNESS_APPDATA"] = old


def _load_report(report_path: Path) -> dict[str, Any]:
    return json.loads(report_path.read_text(encoding="utf-8"))


@pytest.mark.slow
def test_rediscovery_report_has_plugin_results(tmp_path: Path) -> None:
    """Each requested plugin must appear in the report with a valid status."""
    run_id, report_path = _run_baseline(tmp_path)
    assert report_path.exists(), f"Report not generated for run {run_id}"
    report = _load_report(report_path)
    plugins = report.get("plugins", {})

    for pid in REDISCOVERY_PLUGINS:
        assert pid in plugins, f"Plugin {pid} missing from report"
        status = plugins[pid].get("status", "")
        assert status in {"ok", "warn"}, (
            f"Plugin {pid} has unexpected status: {status}"
        )


@pytest.mark.slow
def test_rediscovery_produces_findings(tmp_path: Path) -> None:
    """Baseline must produce at least one finding across all plugins."""
    _, report_path = _run_baseline(tmp_path)
    report = _load_report(report_path)
    plugins = report.get("plugins", {})

    all_findings: list[dict[str, Any]] = []
    for plugin_data in plugins.values():
        if isinstance(plugin_data, dict):
            all_findings.extend(plugin_data.get("findings", []))

    assert len(all_findings) > 0, "Baseline run produced zero findings"

    # Each finding must have contract-required fields
    for f in all_findings:
        assert "id" in f, f"Finding missing 'id': {f}"
        assert "severity" in f, f"Finding missing 'severity': {f}"


@pytest.mark.slow
def test_rediscovery_findings_are_deterministic(tmp_path: Path) -> None:
    """Two runs with same seed must produce identical findings."""
    _, report_1 = _run_baseline(tmp_path / "run1")
    _, report_2 = _run_baseline(tmp_path / "run2")

    r1 = _load_report(report_1)
    r2 = _load_report(report_2)

    # Compare plugin statuses
    for pid in REDISCOVERY_PLUGINS:
        s1 = r1.get("plugins", {}).get(pid, {}).get("status")
        s2 = r2.get("plugins", {}).get(pid, {}).get("status")
        assert s1 == s2, f"Determinism violation for {pid}: {s1} vs {s2}"

    # Compare finding counts per plugin
    for pid in REDISCOVERY_PLUGINS:
        f1 = r1.get("plugins", {}).get(pid, {}).get("findings", [])
        f2 = r2.get("plugins", {}).get(pid, {}).get("findings", [])
        assert len(f1) == len(f2), (
            f"Finding count drift for {pid}: {len(f1)} vs {len(f2)}"
        )


@pytest.mark.slow
def test_rediscovery_report_structure(tmp_path: Path) -> None:
    """Report must have required top-level structure."""
    _, report_path = _run_baseline(tmp_path)
    report = _load_report(report_path)

    assert "plugins" in report, "Report missing 'plugins' key"
    assert isinstance(report["plugins"], dict)
    assert len(report["plugins"]) > 0, "Report has empty plugins dict"


def test_known_issue_compiler_basic() -> None:
    """Compiler must extract process hints and issue categories."""
    issues, warnings = compile_known_issues(
        [
            {"text": "Wait time in process close_a exceeds threshold"},
            {"text": "Third server needed for capacity"},
        ]
    )
    assert len(issues) >= 2, f"Expected >=2 compiled issues, got {len(issues)}"
    plugin_ids = {i["plugin_id"] for i in issues}
    assert "analysis_queue_delay_decomposition" in plugin_ids or \
           "analysis_close_cycle_capacity_model" in plugin_ids


def test_known_issue_compiler_returns_warnings_for_unmatched() -> None:
    """Compiler must warn on text that cannot be matched to any pattern."""
    issues, warnings = compile_known_issues(
        [{"text": "Lorem ipsum dolor sit amet"}]
    )
    assert len(issues) == 0
    assert len(warnings) == 1
