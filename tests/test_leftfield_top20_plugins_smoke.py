from __future__ import annotations

import importlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_context


LEFTFIELD_PLUGIN_IDS = [
    "analysis_ssa_decomposition_changepoint_v1",
    "analysis_cur_decomposition_explain_v1",
    "analysis_hsic_independence_screen_v1",
    "analysis_icp_invariant_causal_prediction_v1",
    "analysis_lingam_causal_discovery_v1",
    "analysis_frequent_directions_cov_sketch_v1",
    "analysis_dmd_koopman_modes_v1",
    "analysis_diffusion_maps_manifold_v1",
    "analysis_sinkhorn_ot_drift_v1",
    "analysis_knn_graph_two_sample_test_v1",
    "analysis_ksd_stein_discrepancy_anomaly_v1",
    "analysis_pc_algorithm_causal_graph_v1",
    "analysis_ges_score_based_causal_v1",
    "analysis_phate_trajectory_embedding_v1",
    "analysis_node2vec_graph_embedding_drift_v1",
    "analysis_tensor_cp_parafac_decomp_v1",
    "analysis_symbolic_regression_gp_v1",
    "analysis_normalizing_flow_density_v1",
    "analysis_tabpfn_foundation_tabular_v1",
    "analysis_neural_additive_model_nam_v1",
]


def test_leftfield_plugin_wrappers_use_dedicated_modules() -> None:
    root = Path(__file__).resolve().parents[1]
    for plugin_id in LEFTFIELD_PLUGIN_IDS:
        wrapper = (root / "plugins" / plugin_id / "plugin.py").read_text(encoding="utf-8")
        assert f"from statistic_harness.core.leftfield_top20.{plugin_id} import run" in wrapper
        assert "leftfield_top20_plugins" not in wrapper


def _build_leftfield_df(rows: int = 420) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    processes = ["qemail", "jbcreateje", "jbinvoice", "postwkfl", "qpec"]
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    out = []
    for i in range(rows):
        t = base + pd.Timedelta(minutes=15 * i)
        p = processes[i % len(processes)]
        wave = math.sin(i / 12.0) + 0.5 * math.cos(i / 7.0)
        drift = 0.002 * i
        v1 = wave + drift + float(rng.normal(0, 0.15))
        v2 = 0.7 * v1 + float(rng.normal(0, 0.18))
        v3 = (v1 * v2) + float(rng.normal(0, 0.08))
        duration = abs(20 + 3 * wave + rng.normal(0, 1.5))
        out.append(
            {
                "PROCESS": p,
                "START_TIME": t.isoformat(),
                "END_TIME": (t + pd.Timedelta(minutes=duration)).isoformat(),
                "VALUE_A": v1,
                "VALUE_B": v2,
                "VALUE_C": v3,
                "DURATION_MIN": duration,
            }
        )
    return pd.DataFrame(out)


@pytest.mark.parametrize("plugin_id", LEFTFIELD_PLUGIN_IDS)
def test_leftfield_plugins_smoke_no_error_no_skip(tmp_path, plugin_id: str) -> None:
    df = _build_leftfield_df()
    ctx = make_context(tmp_path, df, settings={}, run_seed=123)

    from plugins.profile_basic.plugin import Plugin as ProfileBasic
    from plugins.profile_eventlog.plugin import Plugin as ProfileEventlog
    from plugins.transform_normalize_mixed.plugin import Plugin as Normalize

    ctx.settings = {}
    assert ProfileBasic().run(ctx).status == "ok"
    ctx.settings = {}
    assert ProfileEventlog().run(ctx).status == "ok"
    ctx.settings = {"chunk_size": 500}
    assert Normalize().run(ctx).status in {"ok", "degraded"}

    mod = importlib.import_module(f"plugins.{plugin_id}.plugin")
    plugin = mod.Plugin()
    ctx.settings = {}
    result = plugin.run(ctx)
    assert result.status not in {"error", "skipped"}
    assert isinstance(result.metrics, dict)
