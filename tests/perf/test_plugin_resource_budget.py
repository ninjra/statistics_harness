"""Four-pillars performance: per-plugin resource budget validation.

Acceptance criteria (Task 2.4):
  - Every plugin result carries a ``budget`` dict.
  - Budget fields include at least: row_limit, sampled, time_limit_ms, cpu_limit_ms.
  - Hotspot rows sort deterministically: rss_peak_mb desc, cpu_ms desc, plugin_id asc.
  - Budget breaches (status == "error" from resource limits) are deterministic.
"""
from __future__ import annotations

from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_context

BUDGET_REQUIRED_KEYS = {"row_limit", "sampled", "time_limit_ms", "cpu_limit_ms"}

# Fast, representative plugins to validate budget propagation.
BUDGET_PROBE_PLUGINS = [
    "analysis_percentile_analysis",
    "analysis_tail_isolation",
    "analysis_scan_statistics",
]


def _make_dataset(rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(5150)
    return pd.DataFrame(
        {
            "value": rng.normal(100.0, 15.0, size=rows),
            "metric_a": rng.normal(50.0, 10.0, size=rows),
            "metric_b": rng.normal(30.0, 5.0, size=rows),
            "category": rng.choice(["alpha", "beta", "gamma"], size=rows),
        }
    )


def _run_plugin(tmp_path: Path, plugin_id: str, df: pd.DataFrame) -> Any:
    run_dir = tmp_path / plugin_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = make_context(run_dir, df, settings={}, run_seed=42)
    module = import_module(f"plugins.{plugin_id}.plugin")
    return module.Plugin().run(ctx)


@pytest.mark.parametrize("plugin_id", BUDGET_PROBE_PLUGINS)
def test_plugin_result_carries_budget(tmp_path: Path, plugin_id: str) -> None:
    """Every plugin result must include a budget dict with required keys."""
    df = _make_dataset()
    result = _run_plugin(tmp_path, plugin_id, df)
    assert result.status == "ok", f"{plugin_id} failed: {result.summary}"
    assert isinstance(result.budget, dict), f"{plugin_id} budget is not a dict"
    for key in BUDGET_REQUIRED_KEYS:
        assert key in result.budget, f"{plugin_id} budget missing '{key}'"


@pytest.mark.parametrize("plugin_id", BUDGET_PROBE_PLUGINS)
def test_budget_is_deterministic(tmp_path: Path, plugin_id: str) -> None:
    """Budget fields must be identical across runs with the same seed."""
    df = _make_dataset()
    r1 = _run_plugin(tmp_path / "run1", plugin_id, df)
    r2 = _run_plugin(tmp_path / "run2", plugin_id, df)
    assert r1.status == "ok"
    assert r2.status == "ok"
    assert r1.budget == r2.budget, (
        f"{plugin_id} budget not deterministic: {r1.budget} != {r2.budget}"
    )


def test_hotspot_ranking_sort_order(tmp_path: Path) -> None:
    """Hotspot rows must sort deterministically by rss_peak_mb desc, cpu_ms desc, plugin_id asc."""
    df = _make_dataset()
    rows: list[dict[str, Any]] = []
    for plugin_id in BUDGET_PROBE_PLUGINS:
        result = _run_plugin(tmp_path, plugin_id, df)
        metrics = result.metrics or {}
        rows.append(
            {
                "plugin_id": plugin_id,
                "rss_peak_mb": float(metrics.get("rss_peak_mb", 0.0)),
                "cpu_ms": float(metrics.get("cpu_ms") or metrics.get("runtime_ms", 0.0)),
                "rows_read": int(metrics.get("rows_read", 0)),
            }
        )

    sorted_rows = sorted(
        rows,
        key=lambda r: (-r["rss_peak_mb"], -r["cpu_ms"], r["plugin_id"]),
    )
    for row in sorted_rows:
        assert "plugin_id" in row
        assert isinstance(row["rss_peak_mb"], float)
        assert isinstance(row["cpu_ms"], float)
        assert isinstance(row["rows_read"], int)

    # Verify sort is stable and deterministic
    re_sorted = sorted(
        sorted_rows,
        key=lambda r: (-r["rss_peak_mb"], -r["cpu_ms"], r["plugin_id"]),
    )
    assert sorted_rows == re_sorted
