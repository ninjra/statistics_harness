from __future__ import annotations

from typing import Any


def modeled_touch_reduction(
    *,
    run_count: Any,
    top_user_share: Any = None,
) -> float | None:
    if not isinstance(run_count, (int, float)):
        return None
    runs = float(run_count)
    if runs <= 0.0:
        return None
    if isinstance(top_user_share, (int, float)):
        share = max(0.0, min(1.0, float(top_user_share)))
        return runs * share
    return runs


def effort_score(
    *,
    close_hours_saved: Any,
    touches_reduced: Any,
) -> float:
    h = float(close_hours_saved) if isinstance(close_hours_saved, (int, float)) else 0.0
    t = float(touches_reduced) if isinstance(touches_reduced, (int, float)) else 0.0
    if h < 0.0:
        h = 0.0
    if t < 0.0:
        t = 0.0
    return (h * 1.0) + (t ** 0.5)

