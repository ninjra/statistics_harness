"""Smoke tests for cross-domain earth_space plugins."""
import pandas as pd
import numpy as np
import pytest
from statistic_harness.core.stat_plugins.cross_domain_earth_space import HANDLERS


def _make_ctx(df):
    """Minimal mock context for handler testing."""
    from pathlib import Path
    from dataclasses import dataclass, field
    from typing import Any

    @dataclass
    class MockCtx:
        run_id: str = "test"
        run_dir: Path = Path("/tmp/test_run")
        settings: dict = field(default_factory=dict)
        run_seed: int = 42
        budget: dict = field(default_factory=lambda: {
            "row_limit": None, "sampled": False,
            "time_limit_ms": 30000, "cpu_limit_ms": None, "batch_size": None,
        })
        _df: Any = None

        def logger(self, msg): pass
        def dataset_loader(self): return self._df
        def artifacts_dir(self, plugin_id):
            p = self.run_dir / "artifacts" / plugin_id
            p.mkdir(parents=True, exist_ok=True)
            return p

    return MockCtx(_df=df)


def _make_df(n=200):
    """Generate synthetic process log data."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "step_name": rng.choice(["STEP_A", "STEP_B", "STEP_C", "STEP_D"], n),
        "host": rng.choice(["HOST1", "HOST2", "HOST3"], n),
        "duration_sec": rng.exponential(60, n),
        "parent_process_id": [None] + list(range(n - 1)),
        "param_x": rng.uniform(0, 100, n),
        "param_y": rng.choice([0, 1], n),
        "status": rng.choice(["OK", "FAIL"], n, p=[0.9, 0.1]),
    })


def test_all_handlers_registered():
    assert len(HANDLERS) > 0


def test_handlers_return_plugin_result():
    from statistic_harness.core.stat_plugins import BudgetTimer, merge_config, infer_columns

    df = _make_df()
    ctx = _make_ctx(df)
    config = merge_config(ctx.settings)
    inferred = infer_columns(df, config)
    timer = BudgetTimer(30000)
    sample_meta = {"rows_total": len(df), "rows_used": len(df), "sampled": False}

    for plugin_id, handler in list(HANDLERS.items())[:5]:
        result = handler(plugin_id, ctx, df, config, inferred, timer, sample_meta)
        assert result.status in ("ok", "skipped", "degraded", "error", "na"), f"{plugin_id}: bad status {result.status}"
        assert isinstance(result.findings, list), f"{plugin_id}: findings not list"
        assert isinstance(result.metrics, dict), f"{plugin_id}: metrics not dict"


def test_all_handlers_callable():
    from statistic_harness.core.stat_plugins import BudgetTimer, merge_config, infer_columns

    df = _make_df()
    ctx = _make_ctx(df)
    config = merge_config(ctx.settings)
    inferred = infer_columns(df, config)
    sample_meta = {"rows_total": len(df), "rows_used": len(df), "sampled": False}

    for plugin_id, handler in HANDLERS.items():
        timer = BudgetTimer(30000)
        try:
            result = handler(plugin_id, ctx, df, config, inferred, timer, sample_meta)
            assert result.status in ("ok", "skipped", "degraded", "error", "na"), f"{plugin_id}: bad status {result.status}"
        except Exception as e:
            pytest.fail(f"{plugin_id} raised {type(e).__name__}: {e}")
