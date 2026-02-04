from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class CloseWindow:
    accounting_month: str | None
    default_start: Any
    default_end: Any
    dynamic_start: Any
    dynamic_end: Any
    delta_days: float | None


def load_close_cycle_windows(
    run_dir: Path, plugin_id: str = "analysis_close_cycle_window_resolver"
) -> list[CloseWindow]:
    path = run_dir / "artifacts" / plugin_id / "close_windows.csv"
    if not path.exists():
        return []
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []

    for col in (
        "close_start_default",
        "close_end_default",
        "close_start_dynamic",
        "close_end_dynamic",
    ):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    windows: list[CloseWindow] = []
    for row in df.itertuples(index=False):
        windows.append(
            CloseWindow(
                accounting_month=getattr(row, "accounting_month", None),
                default_start=getattr(row, "close_start_default", None),
                default_end=getattr(row, "close_end_default", None),
                dynamic_start=getattr(row, "close_start_dynamic", None),
                dynamic_end=getattr(row, "close_end_dynamic", None),
                delta_days=_safe_float(getattr(row, "close_end_delta_days", None)),
            )
        )
    return windows


def resolve_close_cycle_masks(
    timestamps: Any,
    run_dir: Path,
    close_start_day: int,
    close_end_day: int,
    plugin_id: str = "analysis_close_cycle_window_resolver",
) -> tuple[Any, Any, bool, list[CloseWindow]]:
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        return None, None, False, []
    if timestamps is None:
        return None, None, False, []
    series = pd.to_datetime(timestamps, errors="coerce")
    if hasattr(series, "dropna"):
        series = series
    if series is None or len(series) == 0:
        return None, None, False, []

    default_mask = _mask_from_days(series, close_start_day, close_end_day)
    windows = load_close_cycle_windows(run_dir, plugin_id=plugin_id)
    dynamic_mask = _mask_from_windows(series, windows, "dynamic_start", "dynamic_end")
    if dynamic_mask is None:
        return default_mask, default_mask, False, windows
    return default_mask, dynamic_mask, True, windows


def _mask_from_days(series: Any, start_day: int, end_day: int) -> Any:
    days = series.dt.day
    if start_day <= end_day:
        return (days >= start_day) & (days <= end_day)
    return (days >= start_day) | (days <= end_day)


def _mask_from_windows(
    series: Any,
    windows: Iterable[CloseWindow],
    start_attr: str,
    end_attr: str,
) -> Any:
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        return None
    if not windows:
        return None
    mask = pd.Series(False, index=series.index)
    for window in windows:
        start = getattr(window, start_attr, None)
        end = getattr(window, end_attr, None)
        if start is None or end is None:
            continue
        mask |= (series >= start) & (series <= end)
    return mask


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
