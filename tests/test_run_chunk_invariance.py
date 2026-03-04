from __future__ import annotations

from scripts.run_chunk_invariance import _compare_signature


def test_compare_signature_within_tolerance() -> None:
    baseline = {
        "recommendation_count": 10,
        "total_modeled_delta_hours": 5.0,
        "total_modeled_delta_hours_close_cycle": 3.0,
        "avg_efficiency_gain_pct_close_cycle": 12.0,
        "status_counts": {"ok": 5},
    }
    candidate = {
        "recommendation_count": 10,
        "total_modeled_delta_hours": 5.005,
        "total_modeled_delta_hours_close_cycle": 3.009,
        "avg_efficiency_gain_pct_close_cycle": 12.0005,
        "status_counts": {"ok": 5},
    }
    ok, errors = _compare_signature(
        baseline,
        candidate,
        {"hours_abs": 0.01, "percent_abs": 0.001, "count_abs": 0},
    )
    assert ok is True
    assert errors == []


def test_compare_signature_detects_drift() -> None:
    baseline = {
        "recommendation_count": 10,
        "total_modeled_delta_hours": 5.0,
        "total_modeled_delta_hours_close_cycle": 3.0,
        "avg_efficiency_gain_pct_close_cycle": 12.0,
        "status_counts": {"ok": 5},
    }
    candidate = {
        "recommendation_count": 11,
        "total_modeled_delta_hours": 6.0,
        "total_modeled_delta_hours_close_cycle": 4.0,
        "avg_efficiency_gain_pct_close_cycle": 14.0,
        "status_counts": {"ok": 4, "error": 1},
    }
    ok, errors = _compare_signature(
        baseline,
        candidate,
        {"hours_abs": 0.01, "percent_abs": 0.001, "count_abs": 0},
    )
    assert ok is False
    assert "RECOMMENDATION_COUNT_DRIFT" in errors

