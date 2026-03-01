"""Lightweight PluginContext construction without SQLite dependency.

Uses dataset_loader=lambda: df to bypass Storage/DatasetAccessor entirely.
Suitable for calling run_plugin() from registry or Plugin().run(ctx) directly.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from statistic_harness.core.types import PluginContext


def make_minimal_context(
    df: pd.DataFrame,
    run_dir: Path | None = None,
    run_seed: int = 42,
    settings: dict[str, Any] | None = None,
) -> PluginContext:
    """Create a lightweight PluginContext for verification.

    Skips the active close-cycle window filter to avoid needing run_dir
    artifacts from prior pipeline stages.
    """
    if run_dir is None:
        run_dir = Path(tempfile.mkdtemp(prefix="verify_"))
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    merged_settings: dict[str, Any] = {
        "use_active_close_window_filter": False,
    }
    if settings:
        merged_settings.update(settings)

    frame = df.copy()

    return PluginContext(
        run_id="verification",
        run_dir=run_dir,
        settings=merged_settings,
        run_seed=run_seed,
        logger=lambda msg: None,
        storage=None,
        dataset_loader=lambda: frame.copy(),
    )
