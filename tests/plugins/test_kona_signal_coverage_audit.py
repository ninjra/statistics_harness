from unittest.mock import MagicMock
import pandas as pd
from plugins.analysis_kona_signal_coverage_audit_v1.plugin import Plugin
from tests.conftest import make_context

def test_signal_coverage_audit_with_results(run_dir):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    ctx = make_context(run_dir, df, {})
    ctx.storage.fetch_plugin_results = MagicMock(return_value=[
        {"status": "ok", "findings_json": '[{"kind": "anomaly"}, {"kind": "causal"}, {"kind": "novel_kind"}]'},
        {"status": "ok", "findings_json": '[{"kind": "causal"}]'},
    ])
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.metrics["total_finding_kinds"] >= 2
    assert result.findings[0]["type"] == "signal_coverage_audit"

def test_signal_coverage_audit_no_results(run_dir):
    df = pd.DataFrame({"a": [1.0]})
    ctx = make_context(run_dir, df, {})
    ctx.storage.fetch_plugin_results = MagicMock(return_value=[])
    result = Plugin().run(ctx)
    assert result.status == "ok"
