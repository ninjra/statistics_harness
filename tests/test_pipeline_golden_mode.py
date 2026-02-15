from __future__ import annotations

from statistic_harness.core import pipeline


def test_golden_mode_defaults_off(monkeypatch) -> None:
    monkeypatch.delenv("STAT_HARNESS_GOLDEN_MODE", raising=False)
    assert pipeline._golden_mode() == "off"


def test_golden_mode_accepts_default(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_GOLDEN_MODE", "default")
    assert pipeline._golden_mode() == "default"


def test_golden_mode_truthy_maps_to_strict(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_GOLDEN_MODE", "1")
    assert pipeline._golden_mode() == "strict"


def test_golden_mode_invalid_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_GOLDEN_MODE", "bad-value")
    assert pipeline._golden_mode() == "off"
