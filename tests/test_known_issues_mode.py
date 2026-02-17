from __future__ import annotations

from pathlib import Path

from statistic_harness.core.known_issues_mode import known_issues_enabled, known_issues_mode_label
from statistic_harness.core.report_v2_utils import load_known_issues


def test_known_issues_enabled_defaults_on(monkeypatch) -> None:
    monkeypatch.delenv("STAT_HARNESS_KNOWN_ISSUES_MODE", raising=False)
    assert known_issues_enabled()
    assert known_issues_mode_label() == "on"


def test_known_issues_enabled_off_values(monkeypatch) -> None:
    for value in ("off", "false", "0", "disabled", "none", "no"):
        monkeypatch.setenv("STAT_HARNESS_KNOWN_ISSUES_MODE", value)
        assert not known_issues_enabled()
        assert known_issues_mode_label() == "off"


def test_load_known_issues_short_circuits_when_disabled(monkeypatch, tmp_path: Path) -> None:
    class ExplodingStorage:
        def __getattr__(self, _: str):  # pragma: no cover - this path should never execute
            raise AssertionError("storage should not be accessed when known issues are disabled")

    monkeypatch.setenv("STAT_HARNESS_KNOWN_ISSUES_MODE", "off")
    assert load_known_issues(ExplodingStorage(), "run-1", tmp_path) is None

