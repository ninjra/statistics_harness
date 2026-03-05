"""Tests for Phase 6C: Compositional Lever Generation."""

from __future__ import annotations

from statistic_harness.core.lever_library import LeverRecommendation, compose_levers


def _fake_lever(lever_id: str) -> LeverRecommendation:
    return LeverRecommendation(
        lever_id=lever_id,
        title=f"Fake {lever_id}",
        action=f"Do {lever_id}",
        estimated_improvement_pct=None,
        confidence=0.5,
        evidence={},
        limitations=[],
    )


def test_composition_fires_when_prereqs_met():
    base = [_fake_lever("split_batches"), _fake_lever("blackout_scheduled_jobs")]
    composites = compose_levers(base)
    assert len(composites) == 1
    assert composites[0].lever_id == "split_batches_during_close_window"


def test_composition_does_not_fire_on_partial_prereqs():
    base = [_fake_lever("split_batches")]
    composites = compose_levers(base)
    assert composites == []


def test_multiple_compositions_fire():
    base = [
        _fake_lever("cap_concurrency"),
        _fake_lever("priority_isolation"),
        _fake_lever("retry_backoff"),
        _fake_lever("resource_affinity"),
    ]
    composites = compose_levers(base)
    ids = {c.lever_id for c in composites}
    assert "throttle_with_priority_lanes" in ids
    assert "circuit_breaker_with_cap" in ids
    assert len(composites) == 2
