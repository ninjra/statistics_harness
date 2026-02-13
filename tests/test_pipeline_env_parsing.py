from __future__ import annotations

from statistic_harness.core.pipeline import Pipeline


def test_parse_int_env_returns_none_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("STAT_HARNESS_TEST_INT", raising=False)
    assert Pipeline._parse_int_env("STAT_HARNESS_TEST_INT") is None


def test_parse_int_env_parses_valid_int(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_TEST_INT", "42")
    assert Pipeline._parse_int_env("STAT_HARNESS_TEST_INT") == 42


def test_parse_int_env_returns_none_on_invalid_int(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_TEST_INT", "forty-two")
    assert Pipeline._parse_int_env("STAT_HARNESS_TEST_INT") is None
