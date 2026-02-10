from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LargeDatasetCaps:
    """Harness-level caps for large datasets.

    These are *complexity* caps (not row sampling). Plugins should still scan all rows,
    but must avoid O(n^2) blowups by bounding columns/pairs/groups/windows.
    """

    batch_size: int
    # The harness should not force column/pair/group/window reductions by default.
    # Plugins may still apply their own filtering/selection as needed.
    max_cols: int | None
    max_pairs: int | None
    max_groups: int | None
    max_windows: int | None
    max_findings: int | None
    time_limit_ms: int | None
    cpu_limit_ms: int | None


def caps_for(
    *,
    plugin_id: str,
    plugin_type: str,
    row_count: int | None,
    column_count: int | None,
) -> LargeDatasetCaps | None:
    rc = int(row_count or 0)
    cc = int(column_count or 0)

    # Only engage policy for genuinely large datasets.
    if rc < 1_000_000:
        return None

    # Conservative defaults: cap peak memory per batch, but avoid forcing dimensional reductions.
    batch_size = 100_000
    max_cols = None
    max_pairs = None
    max_groups = None
    max_windows = None
    max_findings = None

    # Transforms should be allowed to run longer; analyses get bounded.
    if plugin_type == "transform":
        time_limit_ms = None
        cpu_limit_ms = None
    else:
        time_limit_ms = None
        cpu_limit_ms = None

    # If dataset is very wide, reduce batch size to cap peak memory.
    if cc >= 200:
        batch_size = 25_000
    elif cc >= 100:
        batch_size = 50_000

    return LargeDatasetCaps(
        batch_size=int(batch_size),
        max_cols=(int(max_cols) if max_cols is not None else None),
        max_pairs=(int(max_pairs) if max_pairs is not None else None),
        max_groups=(int(max_groups) if max_groups is not None else None),
        max_windows=(int(max_windows) if max_windows is not None else None),
        max_findings=(int(max_findings) if max_findings is not None else None),
        time_limit_ms=time_limit_ms,
        cpu_limit_ms=cpu_limit_ms,
    )


def as_budget_dict(caps: LargeDatasetCaps) -> dict[str, Any]:
    return {
        "batch_size": int(caps.batch_size),
        "max_cols": (int(caps.max_cols) if caps.max_cols is not None else None),
        "max_pairs": (int(caps.max_pairs) if caps.max_pairs is not None else None),
        "max_groups": (int(caps.max_groups) if caps.max_groups is not None else None),
        "max_windows": (int(caps.max_windows) if caps.max_windows is not None else None),
        "max_findings": (int(caps.max_findings) if caps.max_findings is not None else None),
        "time_limit_ms": caps.time_limit_ms,
        "cpu_limit_ms": caps.cpu_limit_ms,
        # Explicit: row sampling is not performed by this policy.
        "row_limit": None,
        "sampled": False,
    }
