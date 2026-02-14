from __future__ import annotations

from pathlib import Path

import pandas as pd

from statistic_harness.core.types import PluginResult
from statistic_harness.core.stat_plugins import registry
from statistic_harness.core.stat_plugins.registry import run_plugin
from tests.conftest import make_context


def _write_close_windows_csv(run_dir: Path) -> None:
    artifact_dir = run_dir / "artifacts" / "analysis_close_cycle_window_resolver"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "close_windows.csv"
    pd.DataFrame(
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
        ]
    ).to_csv(path, index=False)


def test_stat_registry_applies_dynamic_active_row_filter(run_dir: Path, monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "START_DT": [
                "2025-12-25T12:00:00",
                "2025-12-29T12:00:00",
                "2025-12-30T12:00:00",
                "2026-01-03T12:00:00",
            ],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    ctx = make_context(run_dir, df, settings={}, run_seed=123)
    _write_close_windows_csv(run_dir)

    plugin_id = "analysis_test_active_row_filter_v1"

    def _handler(
        _plugin_id,
        _ctx,
        work_df,
        _config,
        _inferred,
        _timer,
        _sample_meta,
    ) -> PluginResult:
        return PluginResult(
            status="ok",
            summary="ok",
            metrics={"rows_used_by_handler": int(len(work_df))},
            findings=[],
            artifacts=[],
        )

    monkeypatch.setitem(registry.HANDLERS, plugin_id, _handler)
    result = run_plugin(plugin_id, ctx)

    assert result.status == "ok"
    assert int(result.metrics.get("rows_used_by_handler") or 0) == 2
    meta = result.debug.get("active_row_filter") or {}
    assert meta.get("applied") is True
    assert meta.get("source_plugin") == "analysis_close_cycle_window_resolver"
    assert int(meta.get("rows_before") or 0) == 4
    assert int(meta.get("rows_after") or 0) == 2
