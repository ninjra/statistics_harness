from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_WEIGHTS = {
    "modeled_delta_hours_close_dynamic": 0.40,
    "modeled_delta_hours": 0.25,
    "manual_touch_reduction_count": 0.20,
    "close_contention_reduction_pct": 0.15,
}


def load_weights(path: Path) -> dict[str, float]:
    if not path.exists():
        return dict(DEFAULT_WEIGHTS)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_WEIGHTS)
    if not isinstance(payload, dict):
        return dict(DEFAULT_WEIGHTS)
    block = payload.get("weights")
    if not isinstance(block, dict):
        return dict(DEFAULT_WEIGHTS)
    out = dict(DEFAULT_WEIGHTS)
    for key, value in block.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def weighted_score(item: dict[str, Any], weights: dict[str, float]) -> float:
    score = 0.0
    for key, weight in weights.items():
        raw = item.get(key)
        if not isinstance(raw, (int, float)):
            continue
        score += float(raw) * float(weight)
    return float(score)

