from __future__ import annotations

import numpy as np
import pandas as pd

import statistic_harness.core.stat_plugins.registry as registry_module
from statistic_harness.core.stat_plugins.registry import run_plugin
from tests.conftest import make_context


def test_bsts_counterfactual_hourly_resample_no_error(run_dir) -> None:
    n = 96
    ts = pd.date_range("2026-01-01", periods=n, freq="h")
    values = np.linspace(10.0, 30.0, num=n)
    df = pd.DataFrame(
        {
            "QUEUE_DT": ts.astype(str),
            "VALUE_NUM": values,
        }
    )
    ctx = make_context(run_dir, df, settings={"seed": 7}, run_seed=7)
    res = run_plugin("analysis_bsts_intervention_counterfactual_v1", ctx)
    assert res.status == "ok"
    assert "failed" not in str(res.summary).lower()


def test_bootstrap_ci_tiny_time_budget_returns_ok_not_error(run_dir) -> None:
    n = 400
    ts = pd.date_range("2026-01-01", periods=n, freq="min")
    rng = np.random.default_rng(123)
    df = pd.DataFrame(
        {
            "QUEUE_DT": ts.astype(str),
            "METRIC_A": rng.normal(loc=5.0, scale=1.0, size=n),
            "METRIC_B": rng.normal(loc=6.0, scale=1.5, size=n),
        }
    )
    ctx = make_context(
        run_dir,
        df,
        settings={"seed": 123, "time_budget_ms": 1, "plugin": {"max_resamples": 500}},
        run_seed=123,
    )
    res = run_plugin("analysis_bootstrap_ci_effect_sizes_v1", ctx)
    assert res.status == "ok"
    assert "failed" not in str(res.summary).lower()


def test_change_score_consensus_large_series_no_error(run_dir) -> None:
    n = 20000
    ts = pd.date_range("2026-01-01", periods=n, freq="min")
    rng = np.random.default_rng(99)
    values = rng.normal(loc=0.0, scale=1.0, size=n)
    values[n // 2 :] += 2.0
    df = pd.DataFrame({"START_DT": ts.astype(str), "WAIT_SEC": values})
    ctx = make_context(
        run_dir,
        df,
        settings={"seed": 99, "time_budget_ms": 2000, "plugin": {"max_points": 1200}},
        run_seed=99,
    )
    res = run_plugin("analysis_change_score_consensus_v1", ctx)
    assert res.status == "ok"
    assert "failed" not in str(res.summary).lower()


def test_wild_binary_segmentation_tiny_budget_returns_ok_not_error(run_dir) -> None:
    n = 12000
    ts = pd.date_range("2026-01-01", periods=n, freq="min")
    rng = np.random.default_rng(321)
    values = rng.normal(loc=5.0, scale=0.8, size=n)
    values[8000:] += 1.5
    df = pd.DataFrame({"START_DT": ts.astype(str), "DURATION_SEC": values})
    ctx = make_context(
        run_dir,
        df,
        settings={"seed": 321, "time_budget_ms": 1, "plugin": {"max_points": 800, "intervals": 64}},
        run_seed=321,
    )
    res = run_plugin("analysis_wild_binary_segmentation_v1", ctx)
    assert res.status == "ok"
    assert "failed" not in str(res.summary).lower()


def test_registry_maps_time_budget_exception_to_na(run_dir, monkeypatch) -> None:
    df = pd.DataFrame({"VALUE": [1.0, 2.0, 3.0, 4.0]})
    ctx = make_context(run_dir, df, settings={"seed": 1}, run_seed=1)
    plugin_id = "analysis_timeout_guard_test_v1"

    def _raise_timeout(_plugin_id, _ctx, _df, _config, _inferred, _timer, _sample_meta):
        raise RuntimeError("time_budget_exceeded")

    monkeypatch.setitem(registry_module.HANDLERS, plugin_id, _raise_timeout)
    res = run_plugin(plugin_id, ctx)
    assert res.status == "na"
    assert "time_budget_exceeded" in str(res.summary)
    assert res.findings and res.findings[0].get("kind") == "plugin_not_applicable"


def test_graph_motif_mixed_time_types_no_type_compare_error(run_dir) -> None:
    n = 120
    base = pd.date_range("2026-01-01", periods=n, freq="min").astype(str).tolist()
    mixed_time = [base[i] if i % 7 else float(i) for i in range(n)]
    src = [f"S{i % 9}" for i in range(n)]
    dst = [f"T{(i + 1) % 9}" for i in range(n)]
    df = pd.DataFrame({"START_DT": mixed_time, "SRC_NODE": src, "DST_NODE": dst})
    ctx = make_context(run_dir, df, settings={"seed": 77}, run_seed=77)
    res = run_plugin("analysis_graph_motif_triads_shift_v1", ctx)
    assert res.status == "ok"
    assert "failed" not in str(res.summary).lower()


def test_graph_pagerank_mixed_time_types_no_type_compare_error(run_dir) -> None:
    n = 120
    base = pd.date_range("2026-01-01", periods=n, freq="min").astype(str).tolist()
    mixed_time = [base[i] if i % 5 else float(i) for i in range(n)]
    src = [f"S{i % 7}" for i in range(n)]
    dst = [f"T{(i + 2) % 7}" for i in range(n)]
    df = pd.DataFrame({"START_DT": mixed_time, "SRC_NODE": src, "DST_NODE": dst})
    ctx = make_context(run_dir, df, settings={"seed": 88}, run_seed=88)
    res = run_plugin("analysis_graph_pagerank_hotspots_v1", ctx)
    assert res.status == "ok"
    assert "failed" not in str(res.summary).lower()


def test_dependency_critical_path_mixed_id_types_no_lower_error(run_dir) -> None:
    n = 240
    proc_ids = [str(i) if i % 9 else float(i) for i in range(1, n + 1)]
    parent_ids = [str(i - 1) if i % 5 else float(i - 1) for i in range(1, n + 1)]
    df = pd.DataFrame(
        {
            "PROCESS_QUEUE_ID": proc_ids,
            "PARENT_PROCESS_QUEUE_ID": parent_ids,
            "START_DT": pd.date_range("2026-01-01", periods=n, freq="min").astype(str),
        }
    )
    ctx = make_context(run_dir, df, settings={"seed": 19}, run_seed=19)
    res = run_plugin("analysis_dependency_critical_path_v1", ctx)
    assert res.status != "error"
    assert "failed" not in str(res.summary).lower()


def test_graphical_lasso_ill_conditioned_returns_non_error(run_dir) -> None:
    n = 500
    base = np.linspace(0.0, 1.0, n)
    df = pd.DataFrame(
        {
            "START_DT": pd.date_range("2026-01-01", periods=n, freq="min").astype(str),
            "COL_A": base,
            "COL_B": base,
            "COL_C": base * 2.0,
            "COL_D": base * 3.0,
        }
    )
    ctx = make_context(run_dir, df, settings={"seed": 23}, run_seed=23)
    res = run_plugin("analysis_graphical_lasso_dependency_network", ctx)
    assert res.status in {"ok", "na"}
    assert "failed" not in str(res.summary).lower()


def test_kernel_mmd_non_finite_input_returns_non_error(run_dir) -> None:
    n = 500
    vals = np.linspace(1.0, 10.0, n)
    vals[::11] = np.inf
    vals[::17] = np.nan
    df = pd.DataFrame(
        {
            "START_DT": pd.date_range("2026-01-01", periods=n, freq="min").astype(str),
            "VALUE_A": vals,
            "VALUE_B": vals * 0.5,
        }
    )
    ctx = make_context(run_dir, df, settings={"seed": 31}, run_seed=31)
    res = run_plugin("analysis_kernel_two_sample_mmd", ctx)
    assert res.status in {"ok", "na"}
    assert "failed" not in str(res.summary).lower()
