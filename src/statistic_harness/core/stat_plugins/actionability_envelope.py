from __future__ import annotations

from typing import Any


def non_actionable_envelope(
    *,
    plugin_id: str,
    reason_code: str,
    recommendation: str,
    windows: dict[str, dict[str, float | None]],
    downstream_dependencies: list[str] | None = None,
) -> dict[str, Any]:
    deps = [str(item) for item in (downstream_dependencies or []) if str(item).strip()]
    return {
        "plugin_id": str(plugin_id),
        "status": "non_actionable",
        "reason_code": str(reason_code),
        "recommendation": str(recommendation),
        "windows": windows,
        "downstream_dependencies": sorted(set(deps)),
    }

