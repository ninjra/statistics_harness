from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionCandidate:
    action_type: str
    title: str
    process_norm: str
    delta_seconds: float
    evidence: dict[str, Any]


def clamp_delta_seconds(delta_seconds: float, *, max_hours: float = 5000.0) -> float:
    """Clamp extremely large deltas to keep ranking numerically stable in planners."""

    try:
        val = float(delta_seconds)
    except Exception:
        return 0.0
    cap = float(max_hours) * 3600.0
    if val < 0.0:
        return 0.0
    return min(val, cap)

