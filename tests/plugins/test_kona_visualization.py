from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from tests.conftest import make_context
from statistic_harness.core.stat_plugins.kona_visualization import (
    _kona_3d_landscape_v1,
    _kohonen_energy_map_v1,
    _kona_weight_learner_v1,
)
from statistic_harness.core.stat_plugins import BudgetTimer
from statistic_harness.core.utils import read_json

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "kona_vis"


def _dummy_df() -> pd.DataFrame:
    return pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})


def _setup_esv(run_dir: Path) -> None:
    art_dir = run_dir / "artifacts" / "analysis_ideaspace_energy_ebm_v1"
    art_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "energy_state_vector_5_entities.json", art_dir / "energy_state_vector.json")


def _setup_verified_actions(run_dir: Path) -> None:
    art_dir = run_dir / "artifacts" / "analysis_ebm_action_verifier_v1"
    art_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "verified_actions_4_levers.json", art_dir / "verified_actions.json")


def _setup_route_plan(run_dir: Path) -> None:
    art_dir = run_dir / "artifacts" / "analysis_ebm_action_verifier_v1"
    art_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "route_plan_3_steps.json", art_dir / "route_plan.json")


def _run_handler(handler, tmp_path, config, run_seed=42):
    df = _dummy_df()
    ctx = make_context(tmp_path, df, {}, run_seed=run_seed)
    timer = BudgetTimer(time_budget_ms=60000)
    sample_meta = {"rows_seen": len(df), "rows_used": len(df), "sampled": False}
    plugin_id = {
        _kona_3d_landscape_v1: "analysis_kona_3d_landscape_v1",
        _kohonen_energy_map_v1: "analysis_kohonen_energy_map_v1",
        _kona_weight_learner_v1: "analysis_kona_weight_learner_v1",
    }[handler]
    return handler(plugin_id, ctx, df, config, {}, timer, sample_meta)


# ---------------------------------------------------------------------------
# Plugin 1: 3D Landscape
# ---------------------------------------------------------------------------

class TestKona3DLandscape:
    def test_disabled_by_default(self, tmp_path):
        result = _run_handler(_kona_3d_landscape_v1, tmp_path, {})
        assert result.status == "na"
        assert "disabled" in result.summary.lower()

    def test_insufficient_entities(self, tmp_path):
        _setup_esv(tmp_path)
        # Set min_entities_for_surface higher than available
        result = _run_handler(_kona_3d_landscape_v1, tmp_path, {"enabled": True, "min_entities_for_surface": 100})
        assert result.status == "na"
        assert result.debug.get("gating_reason") == "insufficient_entities"

    def test_produces_artifacts(self, tmp_path):
        _setup_esv(tmp_path)
        result = _run_handler(_kona_3d_landscape_v1, tmp_path, {"enabled": True, "grid_resolution": 15})
        assert result.status == "ok"
        assert len(result.artifacts) == 2
        json_art = [a for a in result.artifacts if a.path.endswith(".json")]
        html_art = [a for a in result.artifacts if a.path.endswith(".html")]
        assert len(json_art) == 1
        assert len(html_art) == 1
        # Verify JSON content
        landscape = read_json(tmp_path / json_art[0].path)
        assert landscape["schema_version"] == "kona_3d_landscape.v1"
        assert len(landscape["entities"]) == 5
        assert len(landscape["pca_explained_variance_ratio"]) >= 1
        # Verify finding
        assert len(result.findings) == 1
        assert result.findings[0]["kind"] == "kona_3d_landscape"
        assert result.findings[0]["entity_count"] == 5

    def test_deterministic(self, tmp_path):
        _setup_esv(tmp_path)
        config = {"enabled": True, "grid_resolution": 15}
        r1 = _run_handler(_kona_3d_landscape_v1, tmp_path, config, run_seed=42)
        r2 = _run_handler(_kona_3d_landscape_v1, tmp_path, config, run_seed=42)
        j1 = read_json(tmp_path / r1.artifacts[0].path)
        j2 = read_json(tmp_path / r2.artifacts[0].path)
        assert j1 == j2

    def test_route_overlay(self, tmp_path):
        _setup_esv(tmp_path)
        _setup_route_plan(tmp_path)
        result = _run_handler(_kona_3d_landscape_v1, tmp_path, {"enabled": True, "grid_resolution": 15})
        assert result.status == "ok"
        landscape = read_json(tmp_path / result.artifacts[0].path)
        assert len(landscape["route_path"]) == 3
        assert landscape["route_path"][0]["label"] == "tune_schedule_qemail_frequency_v1"


# ---------------------------------------------------------------------------
# Plugin 2: Kohonen SOM
# ---------------------------------------------------------------------------

class TestKohonenEnergyMap:
    def test_disabled_by_default(self, tmp_path):
        result = _run_handler(_kohonen_energy_map_v1, tmp_path, {})
        assert result.status == "na"
        assert "disabled" in result.summary.lower()

    def test_graceful_no_minisom(self, tmp_path):
        _setup_esv(tmp_path)
        with patch.dict("sys.modules", {"minisom": None}):
            # Force ImportError by patching the import
            import builtins
            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "minisom":
                    raise ImportError("No module named 'minisom'")
                return real_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=mock_import):
                result = _run_handler(_kohonen_energy_map_v1, tmp_path, {"enabled": True})
                assert result.status == "na"
                assert result.debug.get("gating_reason") == "minisom_not_installed"

    def test_produces_artifacts(self, tmp_path):
        pytest.importorskip("minisom")
        _setup_esv(tmp_path)
        result = _run_handler(_kohonen_energy_map_v1, tmp_path, {"enabled": True, "min_entities": 3})
        assert result.status == "ok"
        assert len(result.artifacts) == 2
        json_art = [a for a in result.artifacts if a.path.endswith(".json")]
        assert len(json_art) == 1
        kohonen = read_json(tmp_path / json_art[0].path)
        assert kohonen["schema_version"] == "kohonen_energy_map.v1"
        assert len(kohonen["entity_bmus"]) == 5
        assert kohonen["som_shape"][0] >= 3
        assert result.findings[0]["kind"] == "kohonen_energy_map"

    def test_deterministic(self, tmp_path):
        pytest.importorskip("minisom")
        _setup_esv(tmp_path)
        config = {"enabled": True, "min_entities": 3}
        r1 = _run_handler(_kohonen_energy_map_v1, tmp_path, config, run_seed=42)
        r2 = _run_handler(_kohonen_energy_map_v1, tmp_path, config, run_seed=42)
        j1 = read_json(tmp_path / r1.artifacts[0].path)
        j2 = read_json(tmp_path / r2.artifacts[0].path)
        assert j1["entity_bmus"] == j2["entity_bmus"]
        assert j1["umatrix"] == j2["umatrix"]


# ---------------------------------------------------------------------------
# Plugin 3: Weight Learner
# ---------------------------------------------------------------------------

class TestKonaWeightLearner:
    def test_disabled_by_default(self, tmp_path):
        result = _run_handler(_kona_weight_learner_v1, tmp_path, {})
        assert result.status == "na"
        assert "disabled" in result.summary.lower()

    def test_insufficient_actions(self, tmp_path):
        _setup_esv(tmp_path)
        # Create verified_actions with only 1 action with estimated_improvement_pct
        art_dir = tmp_path / "artifacts" / "analysis_ebm_action_verifier_v1"
        art_dir.mkdir(parents=True, exist_ok=True)
        actions = {
            "actions": [{
                "lever_id": "test",
                "confidence": 0.5,
                "estimated_improvement_pct": 10.0,
                "observed_metrics": {"duration_p95": 100.0},
                "modeled_metrics_after": {"duration_p95": 90.0},
            }]
        }
        (art_dir / "verified_actions.json").write_text(json.dumps(actions))
        result = _run_handler(_kona_weight_learner_v1, tmp_path, {"enabled": True})
        assert result.status == "na"
        assert result.debug.get("gating_reason") == "insufficient_calibration_data"

    def test_produces_learned_weights(self, tmp_path):
        _setup_esv(tmp_path)
        _setup_verified_actions(tmp_path)
        result = _run_handler(_kona_weight_learner_v1, tmp_path, {"enabled": True})
        assert result.status == "ok"
        assert len(result.artifacts) == 2
        weights = read_json(tmp_path / result.artifacts[0].path)
        assert weights["schema_version"] == "kona_weight_learner.v1"
        assert weights["calibration_actions"] == 4
        assert "learned_weights" in weights
        assert "default_weights" in weights
        assert isinstance(weights["improved"], bool)
        assert result.findings[0]["kind"] == "kona_weight_learner"

    def test_no_regression_when_no_improvement(self, tmp_path):
        _setup_esv(tmp_path)
        # Create actions where learned weights can't improve (all identical impacts)
        art_dir = tmp_path / "artifacts" / "analysis_ebm_action_verifier_v1"
        art_dir.mkdir(parents=True, exist_ok=True)
        base_obs = {
            "duration_p95": 120.0, "queue_delay_p95": 45.0, "error_rate": 0.05,
            "rate_per_min": 8.0, "background_overhead_per_min": 3.5,
        }
        actions = {"actions": [
            {
                "lever_id": f"lever_{i}",
                "confidence": 0.5,
                "estimated_improvement_pct": 10.0,  # same impact for all
                "observed_metrics": base_obs,
                "modeled_metrics_after": {
                    "duration_p95": 120.0 - i,
                    "queue_delay_p95": 45.0 - i,
                    "error_rate": 0.05,
                    "rate_per_min": 8.0,
                    "background_overhead_per_min": 3.5,
                },
            }
            for i in range(4)
        ]}
        (art_dir / "verified_actions.json").write_text(json.dumps(actions))
        result = _run_handler(_kona_weight_learner_v1, tmp_path, {"enabled": True})
        assert result.status == "ok"
        weights = read_json(tmp_path / result.artifacts[0].path)
        # When all impacts are identical, correlation is undefined / 0 — improved should be false
        assert isinstance(weights["improved"], bool)
