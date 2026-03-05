import pandas as pd
from unittest.mock import MagicMock
from plugins.analysis_causal_cross_validation_v1.plugin import Plugin, _extract_effect
from tests.conftest import make_context


def test_extract_effect_from_metrics():
    data = {"metrics": {"ate": 2.5}, "findings": []}
    assert _extract_effect(data) == 2.5


def test_extract_effect_from_findings():
    data = {"metrics": {}, "findings": [{"kind": "causal", "did_estimate": -1.3}]}
    assert _extract_effect(data) == -1.3


def test_cross_validation_sign_agreement(run_dir):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    ctx = make_context(run_dir, df, {})
    # Mock storage to return synthetic results
    results = {
        "analysis_double_ml_ate_v1": {"status": "ok", "metrics": {"ate": 2.0}, "findings": []},
        "analysis_diff_in_diff_v1": {"status": "ok", "metrics": {"did_estimate": 2.5}, "findings": []},
        "analysis_propensity_score_matching_v1": {"status": "ok", "metrics": {"att": -1.0}, "findings": []},
    }
    ctx.storage.fetch_latest_plugin_result = MagicMock(
        side_effect=lambda run_id, pid: results.get(pid)
    )
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.metrics["n_causal_plugins_with_effects"] == 3
    # Should detect sign disagreement
    sign_findings = [f for f in result.findings if f.get("check") == "sign_disagreement"]
    assert sign_findings


def test_cross_validation_insufficient_plugins(run_dir):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    ctx = make_context(run_dir, df, {})
    ctx.storage.fetch_latest_plugin_result = MagicMock(return_value=None)
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.metrics["n_causal_plugins_with_effects"] == 0
