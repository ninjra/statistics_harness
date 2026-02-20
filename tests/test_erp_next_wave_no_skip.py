from __future__ import annotations

import pandas as pd

from statistic_harness.core.stat_plugins.registry import run_plugin
from tests.conftest import make_context


def test_hold_time_attribution_missing_columns_returns_ok_observation(run_dir):
    df = pd.DataFrame({"metric": [1, 2, 3], "value": [3, 4, 5]})
    ctx = make_context(run_dir, df, {})
    result = run_plugin("analysis_hold_time_attribution_v1", ctx)
    assert result.status == "ok"
    findings = result.findings if isinstance(result.findings, list) else []
    assert findings
    row = findings[0] if isinstance(findings[0], dict) else {}
    assert row.get("kind") == "plugin_observation"
    assert row.get("reason_code") == "MISSING_REQUIRED_COLUMNS"


def test_retry_rate_hotspots_missing_columns_returns_ok_observation(run_dir):
    df = pd.DataFrame({"PROCESS_ID": ["p1", "p2", "p3"], "metric": [1, 2, 3]})
    ctx = make_context(run_dir, df, {})
    result = run_plugin("analysis_retry_rate_hotspots_v1", ctx)
    assert result.status == "ok"
    findings = result.findings if isinstance(result.findings, list) else []
    assert findings
    row = findings[0] if isinstance(findings[0], dict) else {}
    assert row.get("kind") == "plugin_observation"
    assert row.get("reason_code") == "MISSING_REQUIRED_COLUMNS"
