import json

import pandas as pd
import pytest
from pathlib import Path

from plugins.analysis_fragility_perturbation_v1.plugin import Plugin
from tests.conftest import make_context


def test_fragility_perturbation_produces_findings(run_dir):
    # Create mock EBM artifact
    artifact_dir = run_dir / "artifacts" / "analysis_ideaspace_energy_ebm_v1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "energy_keys": ["duration_p50", "duration_p95", "throughput"],
        "weights": {"duration_p50": 1.0, "duration_p95": 1.0, "throughput": 0.5},
        "entities": [
            {
                "entity_key": "ALL",
                "observed": {"duration_p50": 10.0, "duration_p95": 50.0, "throughput": 100.0},
                "ideal": {"duration_p50": 5.0, "duration_p95": 20.0, "throughput": 200.0},
                "energy_total": 0.5,
            }
        ],
    }
    (artifact_dir / "energy_state_vector.json").write_text(json.dumps(state))

    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert len(result.findings) >= 1
    assert result.findings[0]["kind"] == "fragility"
    assert "most_fragile_metric" in result.findings[0]


def test_fragility_skips_without_ebm_artifact(run_dir):
    df = pd.DataFrame({"a": [1.0]})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "skipped"
