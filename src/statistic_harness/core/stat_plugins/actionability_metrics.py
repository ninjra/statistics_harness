from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WindowMetric:
    delta_h: float | None
    eff_pct: float | None
    eff_idx: float | None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "delta_h": self.delta_h,
            "eff_pct": self.eff_pct,
            "eff_idx": self.eff_idx,
        }


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def build_window_metric(delta_h: Any, eff_pct: Any, baseline_hours: Any) -> WindowMetric:
    delta = _num(delta_h)
    pct = _num(eff_pct)
    baseline = _num(baseline_hours)
    eff_idx: float | None = None
    if delta is not None and baseline is not None and baseline > 0:
        eff_idx = round(delta / baseline, 6)
    return WindowMetric(
        delta_h=round(delta, 6) if delta is not None else None,
        eff_pct=round(pct, 6) if pct is not None else None,
        eff_idx=eff_idx,
    )


def build_window_triplet(
    *,
    accounting_month: WindowMetric,
    close_static: WindowMetric,
    close_dynamic: WindowMetric,
) -> dict[str, dict[str, float | None]]:
    return {
        "accounting_month": accounting_month.as_dict(),
        "close_static": close_static.as_dict(),
        "close_dynamic": close_dynamic.as_dict(),
    }

