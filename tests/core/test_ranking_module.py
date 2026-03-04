from __future__ import annotations

from pathlib import Path

from statistic_harness.core.ranking import load_weights, weighted_score


def test_load_weights_defaults_on_missing_file(tmp_path: Path) -> None:
    weights = load_weights(tmp_path / "missing.yaml")
    assert "modeled_delta_hours_close_cycle" in weights
    assert "value_score_v2" in weights


def test_weighted_score() -> None:
    item = {
        "modeled_delta_hours_close_cycle": 2.0,
        "modeled_delta_hours": 1.0,
        "modeled_user_touches_reduced": 10.0,
        "value_score_v2": 0.8,
    }
    score = weighted_score(item, {"modeled_delta_hours_close_cycle": 1.0, "modeled_user_touches_reduced": 0.1})
    assert score == 3.0
