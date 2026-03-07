import json
import pandas as pd
from plugins.analysis_cross_entity_transfer_v1.plugin import Plugin
from tests.conftest import make_context


def test_cross_entity_transfer_finds_opportunity(run_dir):
    artifact_dir = run_dir / "artifacts" / "analysis_ideaspace_energy_ebm_v1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "energy_keys": ["duration_p50", "duration_p95", "throughput"],
        "entities": [
            {
                "entity_key": "ALL",
                "observed": {"duration_p50": 10.0, "duration_p95": 50.0, "throughput": 100.0},
                "ideal": {"duration_p50": 5.0, "duration_p95": 20.0, "throughput": 200.0},
                "energy_total": 0.5,
            },
            {
                "entity_key": "entity_A",
                "observed": {"duration_p50": 8.0, "duration_p95": 40.0, "throughput": 95.0},
                "ideal": {"duration_p50": 5.0, "duration_p95": 20.0, "throughput": 200.0},
                "energy_total": 0.3,
            },
            {
                "entity_key": "entity_B",
                "observed": {"duration_p50": 8.5, "duration_p95": 80.0, "throughput": 98.0},
                "ideal": {"duration_p50": 5.0, "duration_p95": 20.0, "throughput": 200.0},
                "energy_total": 0.7,
            },
        ],
    }
    (artifact_dir / "energy_state_vector.json").write_text(json.dumps(state))

    df = pd.DataFrame({"a": [1.0, 2.0]})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert len(result.findings) >= 1
    assert result.findings[0]["kind"] == "transfer_recommendation"
    assert "transfer_metric" in result.findings[0]


def test_cross_entity_transfer_skips_without_artifact(run_dir):
    df = pd.DataFrame({"a": [1.0]})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "na"


def test_cross_entity_transfer_skips_single_entity(run_dir):
    artifact_dir = run_dir / "artifacts" / "analysis_ideaspace_energy_ebm_v1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "energy_keys": ["duration_p50"],
        "entities": [
            {
                "entity_key": "ALL",
                "observed": {"duration_p50": 10.0},
                "ideal": {"duration_p50": 5.0},
                "energy_total": 0.5,
            },
        ],
    }
    (artifact_dir / "energy_state_vector.json").write_text(json.dumps(state))

    df = pd.DataFrame({"a": [1.0]})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "na"
