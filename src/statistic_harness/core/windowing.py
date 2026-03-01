from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .accounting_windows import (
    AccountingWindow,
    infer_accounting_windows_from_timestamps,
    load_accounting_windows_from_run,
    window_ranges,
)


def resolve_accounting_windows(
    *,
    run_dir: Path | None = None,
    timestamps: Any | None = None,
) -> list[AccountingWindow]:
    """Resolve windows from run artifacts first, then timestamp inference."""

    if run_dir is not None:
        loaded = load_accounting_windows_from_run(run_dir)
        if loaded:
            return loaded
    if timestamps is None:
        return []
    series = pd.to_datetime(timestamps, errors="coerce")
    return infer_accounting_windows_from_timestamps(series)


def window_payload(windows: list[AccountingWindow]) -> dict[str, Any]:
    def _serialize_ranges(kind: str) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for start, end in window_ranges(windows, kind=kind):
            out.append({"start": start.isoformat(), "end": end.isoformat()})
        return out

    return {
        "schema_version": "windowing.v1",
        "count": int(len(windows)),
        "items": [asdict(row) for row in windows],
        "ranges": {
            "accounting_month": _serialize_ranges("accounting_month"),
            "close_static": _serialize_ranges("close_static"),
            "close_dynamic": _serialize_ranges("close_dynamic"),
        },
    }
