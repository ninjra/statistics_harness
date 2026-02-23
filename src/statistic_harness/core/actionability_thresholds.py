from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ActionabilityThresholds:
    delta_hours_accounting_month: float
    eff_pct_accounting_month: float
    eff_idx_accounting_month: float
    confidence: float
    require_all: bool = True
    fallback_status: str = "na"
    fallback_reason_code: str = "BELOW_ACTIONABILITY_THRESHOLD"


_DEFAULT_THRESHOLDS = ActionabilityThresholds(
    delta_hours_accounting_month=0.25,
    eff_pct_accounting_month=2.0,
    eff_idx_accounting_month=0.2,
    confidence=0.15,
    require_all=True,
    fallback_status="na",
    fallback_reason_code="BELOW_ACTIONABILITY_THRESHOLD",
)


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "actionability_thresholds.yaml"


def load_actionability_thresholds(path: Path | None = None) -> ActionabilityThresholds:
    cfg_path = path or _default_config_path()
    payload: dict[str, Any] = {}
    if cfg_path.exists():
        try:
            parsed = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                payload = parsed
        except (OSError, PermissionError, yaml.YAMLError):
            # Plugin sandbox may deny direct config reads; fail closed with
            # deterministic built-in thresholds instead of returning an error.
            payload = {}
    minimums = payload.get("minimums") if isinstance(payload.get("minimums"), dict) else {}
    scoring = payload.get("scoring") if isinstance(payload.get("scoring"), dict) else {}
    return ActionabilityThresholds(
        delta_hours_accounting_month=float(
            minimums.get(
                "delta_hours_accounting_month",
                _DEFAULT_THRESHOLDS.delta_hours_accounting_month,
            )
        ),
        eff_pct_accounting_month=float(
            minimums.get(
                "eff_pct_accounting_month",
                _DEFAULT_THRESHOLDS.eff_pct_accounting_month,
            )
        ),
        eff_idx_accounting_month=float(
            minimums.get(
                "eff_idx_accounting_month",
                _DEFAULT_THRESHOLDS.eff_idx_accounting_month,
            )
        ),
        confidence=float(minimums.get("confidence", _DEFAULT_THRESHOLDS.confidence)),
        require_all=bool(scoring.get("require_all", _DEFAULT_THRESHOLDS.require_all)),
        fallback_status=str(scoring.get("fallback_status") or _DEFAULT_THRESHOLDS.fallback_status),
        fallback_reason_code=str(
            scoring.get("fallback_reason_code") or _DEFAULT_THRESHOLDS.fallback_reason_code
        ),
    )


def meets_actionability_thresholds(
    delta_hours_accounting_month: float,
    eff_pct_accounting_month: float,
    eff_idx_accounting_month: float,
    confidence: float,
    *,
    thresholds: ActionabilityThresholds | None = None,
) -> bool:
    t = thresholds or load_actionability_thresholds()
    checks = [
        float(delta_hours_accounting_month) >= t.delta_hours_accounting_month,
        float(eff_pct_accounting_month) >= t.eff_pct_accounting_month,
        float(eff_idx_accounting_month) >= t.eff_idx_accounting_month,
        float(confidence) >= t.confidence,
    ]
    return all(checks) if t.require_all else any(checks)
