from __future__ import annotations

from statistic_harness.core.report import _apply_recency_weight


def test_recency_weighting_uses_configured_decay_and_floor(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_RECENCY_DECAY_PER_MONTH", "1.0")
    monkeypatch.setenv("STAT_HARNESS_RECENCY_MIN_WEIGHT", "0.2")

    items = [
        {"close_month": "2026-02", "relevance_score": 10.0},
        {"close_month": "2025-02", "relevance_score": 10.0},
    ]
    weighted = _apply_recency_weight(items)

    assert weighted[0]["recency_weight"] == 1.0
    assert weighted[1]["recency_weight"] == 0.2
    assert weighted[1]["relevance_score"] == 2.0


def test_recency_weighting_uses_configured_unknown_weight(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_RECENCY_UNKNOWN_WEIGHT", "0.73")
    monkeypatch.setenv("STAT_HARNESS_RECENCY_DECAY_PER_MONTH", "0.25")
    monkeypatch.setenv("STAT_HARNESS_RECENCY_MIN_WEIGHT", "0.4")

    items = [
        {"close_month": "2026-02", "relevance_score": 10.0},
        {"recommendation": "missing month", "relevance_score": 10.0},
    ]
    weighted = _apply_recency_weight(items)

    assert weighted[0]["recency_weight"] == 1.0
    assert weighted[1]["recency_weight"] == 0.73
    assert weighted[1]["relevance_score"] == 7.3
