from __future__ import annotations

import importlib

import pandas as pd
import pytest

from tests.conftest import make_context


PLUGIN_IDS = [
    "analysis_elastic_net_regularized_glm_v1",
    "analysis_minimum_covariance_determinant_v1",
    "analysis_gaussian_process_regression_v1",
    "analysis_mixed_effects_hierarchical_v1",
    "analysis_bart_uplift_surrogate_v1",
    "analysis_granger_causality_v1",
    "analysis_nonnegative_matrix_factorization_v1",
    "analysis_tsne_embedding_v1",
    "analysis_umap_embedding_v1",
    "analysis_mice_imputation_chained_equations_v1",
]


def _build_df() -> pd.DataFrame:
    rows = []
    for i in range(120):
        rows.append(
            {
                "PROCESS": "RPT_POR002" if i % 2 == 0 else "QPEC",
                "START_TIME": f"2026-01-{(i % 27) + 1:02d}T00:{i % 60:02d}:00",
                "END_TIME": f"2026-01-{(i % 27) + 1:02d}T00:{(i + 1) % 60:02d}:30",
                "USER_NAME": "user_a" if i % 3 == 0 else "user_b",
                "metric_a": float(i),
                "metric_b": float((i * 2) % 37),
                "metric_c": float((i * 3) % 19),
                "metric_d": float((i * 5) % 23),
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("plugin_id", PLUGIN_IDS)
def test_plugins_validate_runbook_surrogate_smoke(run_dir, plugin_id: str) -> None:
    ctx = make_context(run_dir, _build_df(), settings={})
    module = importlib.import_module(f"plugins.{plugin_id}.plugin")
    plugin = module.Plugin()
    result = plugin.run(ctx)
    assert result.status in {"ok", "na"}
    if result.status == "ok":
        assert any(f.get("kind") == "actionable_ops_lever" for f in result.findings)
        metrics = result.metrics
        assert "delta_h_accounting_month" in metrics
        assert "eff_pct_accounting_month" in metrics
        assert "eff_idx_accounting_month" in metrics
    if result.status == "na":
        assert any("reason_code" in f for f in result.findings)
