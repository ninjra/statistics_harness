from __future__ import annotations

from datetime import datetime
from pathlib import Path

from statistic_harness.core.windowing import resolve_accounting_windows, window_payload


def test_windowing_resolves_from_timestamps() -> None:
    ts = [
        datetime(2026, 1, 3, 10, 0, 0),
        datetime(2026, 1, 20, 10, 0, 0),
        datetime(2026, 2, 4, 10, 0, 0),
    ]
    windows = resolve_accounting_windows(run_dir=None, timestamps=ts)
    assert windows
    payload = window_payload(windows)
    assert payload["count"] >= 1


def test_windowing_empty_without_inputs(tmp_path: Path) -> None:
    windows = resolve_accounting_windows(run_dir=tmp_path / "missing", timestamps=None)
    assert windows == []

