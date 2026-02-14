from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_CLOSE_WINDOW_PLUGIN_IDS: tuple[str, ...] = (
    "analysis_close_cycle_window_resolver",
    "analysis_close_cycle_start_backtrack_v1",
)


@dataclass(frozen=True)
class CloseWindow:
    accounting_month: str | None
    default_start: Any
    default_end: Any
    dynamic_start: Any
    dynamic_end: Any
    delta_days: float | None
    source: str | None = None
    confidence: float | None = None
    fallback_reason: str | None = None


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
                source=getattr(row, "source", None),
                confidence=_safe_float(getattr(row, "confidence", None)),
                fallback_reason=getattr(row, "fallback_reason", None),
            )
        )
    return windows


def load_preferred_close_cycle_windows(
    run_dir: Path,
    plugin_ids: Iterable[str] | None = None,
) -> tuple[list[CloseWindow], str | None]:
    ordered = list(plugin_ids or DEFAULT_CLOSE_WINDOW_PLUGIN_IDS)
    for plugin_id in ordered:
        windows = load_close_cycle_windows(run_dir, plugin_id=plugin_id)
        if windows:
            return windows, plugin_id
    return [], None


def resolve_active_close_cycle_mask(
    timestamps: Any,
    run_dir: Path,
    plugin_ids: Iterable[str] | None = None,
) -> tuple[Any, bool, str | None, list[CloseWindow]]:
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        return None, False, None, []
    if timestamps is None:
        return None, False, None, []
    series = pd.to_datetime(timestamps, errors="coerce")
    if not hasattr(series, "dt"):
        series = pd.Series(series)
    if series is None or len(series) == 0:
        return None, False, None, []

    windows, source_plugin = load_preferred_close_cycle_windows(
        run_dir, plugin_ids=plugin_ids
    )
    if not windows:
        return None, False, source_plugin, []
    mask = _mask_from_windows(series, windows, "dynamic_start", "dynamic_end")
    if mask is None:
        return None, False, source_plugin, windows
    return mask, True, source_plugin, windows


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
    if not hasattr(series, "dt"):
        # Accept DatetimeIndex inputs as well as Series.
        series = pd.Series(series)
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


def compute_close_month(
    timestamps: Any, *, baseline_close_end_day: int
) -> Any:
    """Compute a 'close_month' cohort label (YYYY-MM) using the wrap-end rule.

    Rule: if day-of-month <= baseline_close_end_day, treat it as belonging to the
    previous month (e.g. Jan 5 belongs to Dec close).
    """

    try:
        import pandas as pd  # type: ignore
    except ImportError:
        return None
    series = pd.to_datetime(timestamps, errors="coerce", utc=False)
    if not hasattr(series, "dt"):
        series = pd.Series(series)
    if series is None or len(series) == 0:
        return None
    days = series.dt.day
    shifted = series - pd.DateOffset(months=1)
    cohort = series.dt.to_period("M").astype(str)
    shifted_cohort = shifted.dt.to_period("M").astype(str)
    out = cohort.where(~(days <= int(baseline_close_end_day)), shifted_cohort)
    return out


def baseline_target_spillover_masks(
    timestamps: Any,
    *,
    baseline_close_start_day: int,
    baseline_close_end_day: int,
    target_close_end_day: int,
) -> tuple[Any, Any, Any]:
    """Return (baseline_mask, target_mask, spillover_mask) for the given timestamps."""

    try:
        import pandas as pd  # type: ignore
    except ImportError:
        return None, None, None
    series = pd.to_datetime(timestamps, errors="coerce", utc=False)
    if not hasattr(series, "dt"):
        series = pd.Series(series)
    if series is None or len(series) == 0:
        return None, None, None
    days = series.dt.day
    baseline_mask = _mask_from_days(series, int(baseline_close_start_day), int(baseline_close_end_day))
    # Target close window is defined as start_day -> EOM/target_end (no wrap).
    target_mask = (days >= int(baseline_close_start_day)) & (days <= int(target_close_end_day))
    spillover_mask = baseline_mask & (~target_mask)
    return baseline_mask, target_mask, spillover_mask
