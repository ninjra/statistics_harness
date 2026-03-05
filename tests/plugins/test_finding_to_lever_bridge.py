from unittest.mock import MagicMock
import pandas as pd
from plugins.analysis_finding_to_lever_bridge_v1.plugin import Plugin
from statistic_harness.core.finding_lever_map import match_finding_to_levers
from tests.conftest import make_context

def test_match_finding_to_levers_causal():
    finding = {"kind": "causal", "ate": 2.5}
    matches = match_finding_to_levers(finding)
    assert len(matches) >= 1
    assert matches[0]["lever"].lever_id == "address_causal_treatment_effect"

def test_match_finding_no_match():
    finding = {"kind": "completely_unknown_kind_xyz"}
    matches = match_finding_to_levers(finding)
    assert matches == []

def test_bridge_produces_lever_candidates(run_dir):
    df = pd.DataFrame({"a": [1.0, 2.0]})
    ctx = make_context(run_dir, df, {})
    ctx.storage.fetch_plugin_results = MagicMock(return_value=[
        {"status": "ok", "plugin_id": "p1", "findings_json": '[{"kind": "causal", "ate": 2.0}]'},
        {"status": "ok", "plugin_id": "p2", "findings_json": '[{"kind": "counterfactual", "effect": 1.5}]'},
    ])
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.metrics["lever_candidates"] >= 2

def test_bridge_no_upstream(run_dir):
    df = pd.DataFrame({"a": [1.0]})
    ctx = make_context(run_dir, df, {})
    ctx.storage.fetch_plugin_results = MagicMock(return_value=[])
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.metrics["lever_candidates"] == 0
