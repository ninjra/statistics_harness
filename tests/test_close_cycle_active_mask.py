from __future__ import annotations

from pathlib import Path

import pandas as pd

from statistic_harness.core.close_cycle import (
    load_preferred_close_cycle_windows,
    resolve_active_close_cycle_mask,
)


def _write_close_windows_csv(base: Path, plugin_id: str, rows: list[dict[str, str]]) -> None:
    plugin_dir = base / "artifacts" / plugin_id
    plugin_dir.mkdir(parents=True, exist_ok=True)
    path = plugin_dir / "close_windows.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def test_resolve_active_mask_prefers_resolver_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_close_windows_csv(
        run_dir,
        "analysis_close_cycle_window_resolver",
        [
            {
                "accounting_month": "2026-01",
                "close_start_default": "2025-12-20T00:00:00",
                "close_end_default": "2026-01-05T00:00:00",
                "close_start_dynamic": "2025-12-28T00:00:00",
                "close_end_dynamic": "2026-01-02T23:59:59",
                "close_end_delta_days": "0.0",
                "source": "backtracked_signature",
                "confidence": "0.8",
                "fallback_reason": "",
            }
        ],
    )
    _write_close_windows_csv(
        run_dir,
        "analysis_close_cycle_start_backtrack_v1",
        [
            {
                "accounting_month": "2026-01",
                "close_start_default": "2025-12-20T00:00:00",
                "close_end_default": "2026-01-05T00:00:00",
                "close_start_dynamic": "2025-12-29T00:00:00",
                "close_end_dynamic": "2026-01-03T23:59:59",
                "close_end_delta_days": "0.0",
                "source": "backtracked_signature",
                "confidence": "0.75",
                "fallback_reason": "",
            }
        ],
    )

    windows, source_plugin = load_preferred_close_cycle_windows(run_dir)
    assert source_plugin == "analysis_close_cycle_window_resolver"
    assert len(windows) == 1

    ts = pd.to_datetime(
        [
            "2025-12-27T12:00:00",
            "2025-12-30T12:00:00",
            "2026-01-03T12:00:00",
        ]
    )
    mask, used_dynamic, mask_source, _ = resolve_active_close_cycle_mask(ts, run_dir)
    assert used_dynamic is True
    assert mask_source == "analysis_close_cycle_window_resolver"
    assert mask.tolist() == [False, True, False]


def test_resolve_active_mask_falls_back_to_backtrack_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_close_windows_csv(
        run_dir,
        "analysis_close_cycle_start_backtrack_v1",
        [
            {
                "accounting_month": "2026-02",
                "close_start_default": "2026-01-20T00:00:00",
                "close_end_default": "2026-02-05T00:00:00",
                "close_start_dynamic": "2026-01-27T00:00:00",
                "close_end_dynamic": "2026-02-01T23:59:59",
                "close_end_delta_days": "0.0",
                "source": "backtracked_signature",
                "confidence": "0.7",
                "fallback_reason": "",
            }
        ],
    )

    ts = pd.to_datetime(
        [
            "2026-01-25T12:00:00",
            "2026-01-30T12:00:00",
            "2026-02-03T12:00:00",
        ]
    )
    mask, used_dynamic, source_plugin, windows = resolve_active_close_cycle_mask(ts, run_dir)
    assert used_dynamic is True
    assert source_plugin == "analysis_close_cycle_start_backtrack_v1"
    assert len(windows) == 1
    assert mask.tolist() == [False, True, False]
