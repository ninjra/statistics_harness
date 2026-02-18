import datetime as dt

import pandas as pd

from plugins.analysis_chi_square_association.plugin import Plugin
from statistic_harness.core.stat_plugins import topo_tda_addon
from tests.conftest import make_context


def _sample_df() -> pd.DataFrame:
    rows = 240
    # Includes some PII-shaped strings to ensure redaction paths are exercised.
    return pd.DataFrame(
        {
            "metric": [0.1] * 120 + [2.0] * 120,
            "metric2": [1.0] * 60 + [3.0] * 180,
            "category": ["A"] * 120 + ["B"] * 120,
            "email": ["user@example.com"] * rows,
            "uuid": ["123e4567-e89b-12d3-a456-426614174000"] * rows,
            "case_id": [i // 6 for i in range(rows)],
            "ts": [dt.datetime(2026, 1, 1) + dt.timedelta(minutes=i) for i in range(rows)],
            "x": [float(i % 20) for i in range(rows)],
            "y": [float(i % 12) for i in range(rows)],
        }
    )


def test_analysis_chi_square_association_smoke(run_dir):
    df = _sample_df()
    df["ts"] = df["ts"].astype(str)
    ctx = make_context(run_dir, df, {}, run_seed=1337)
    result = Plugin().run(ctx)
    assert result.status in ("ok", "skipped", "degraded")


def test_analysis_chi_square_association_zero_expected_frequencies_degrades_cleanly(run_dir, monkeypatch):
    rows = 240
    df = pd.DataFrame(
        {
            "cat_a": ["A"] * 120 + ["B"] * 120,
            "cat_b": (["X"] * 60 + ["Y"] * 60) * 2,
            "metric": [1.0] * rows,
            "ts": [dt.datetime(2026, 1, 1) + dt.timedelta(minutes=i) for i in range(rows)],
        }
    )
    df["ts"] = df["ts"].astype(str)
    ctx = make_context(run_dir, df, {}, run_seed=1337)
    calls = {"n": 0}

    def _raise_zero_expected(table):
        calls["n"] += 1
        raise ValueError(
            "The internally computed table of expected frequencies has a zero element at (0, 0)."
        )

    monkeypatch.setattr(topo_tda_addon.scipy_stats, "chi2_contingency", _raise_zero_expected)
    result = Plugin().run(ctx)

    assert result.status == "ok"
    assert calls["n"] > 0


def test_analysis_chi_square_association_crosstab_alloc_guard(run_dir, monkeypatch):
    rows = 240
    df = pd.DataFrame(
        {
            "cat_a": ["A"] * 120 + ["B"] * 120,
            "cat_b": (["X"] * 60 + ["Y"] * 60) * 2,
            "metric": [1.0] * rows,
            "ts": [dt.datetime(2026, 1, 1) + dt.timedelta(minutes=i) for i in range(rows)],
        }
    )
    df["ts"] = df["ts"].astype(str)
    ctx = make_context(run_dir, df, {}, run_seed=1337)
    calls = {"n": 0}

    original_crosstab = topo_tda_addon.pd.crosstab

    def _raise_alloc(*args, **kwargs):
        calls["n"] += 1
        raise ValueError("Unable to allocate 2.36 GiB for an array")

    monkeypatch.setattr(topo_tda_addon.pd, "crosstab", _raise_alloc)
    try:
        result = Plugin().run(ctx)
    finally:
        monkeypatch.setattr(topo_tda_addon.pd, "crosstab", original_crosstab)

    assert calls["n"] > 0
    assert result.status == "ok"
    assert result.metrics.get("skipped_crosstab_alloc", 0) > 0
