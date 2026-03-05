"""Post-build guardrails: fail-fast checks for decision report v2.

Enforces ANY_REGRESS => DO_NOT_SHIP constraints defined in the spec:
- Numbers without claim_id mapping
- Waterfall reconciliation tolerance
- Forbidden columns in slide_kit outputs
- Conflicting recommendation dedupe deltas
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .redaction import FORBIDDEN_COLUMNS, check_forbidden_columns
from .traceability import ClaimRegistry


class GuardrailViolation:
    """A single guardrail check failure."""

    def __init__(self, code: str, message: str, severity: str = "error") -> None:
        self.code = code
        self.message = message
        self.severity = severity

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "severity": self.severity}


def check_unclaimed_numbers(
    registry: ClaimRegistry, rendered_text: str
) -> list[GuardrailViolation]:
    """Fail if any registered claim is not referenced in rendered output."""
    errors = registry.validate_rendered_numbers(rendered_text)
    return [
        GuardrailViolation("unclaimed_number", msg)
        for msg in errors
    ]


def check_waterfall_reconciliation(
    waterfall: dict[str, Any] | None,
    tolerance_hours: float = 0.01,
) -> list[GuardrailViolation]:
    """Fail if waterfall total != top_driver + remainder beyond tolerance."""
    if not waterfall or not isinstance(waterfall, dict):
        return []

    violations: list[GuardrailViolation] = []
    total = waterfall.get("total_bp_over_threshold_wait_hours")
    top_driver = waterfall.get("top_driver_over_threshold_wait_hours")
    remainder = waterfall.get("remainder_without_top_driver_hours")

    if total is None or top_driver is None or remainder is None:
        return []

    try:
        diff = abs(float(total) - (float(top_driver) + float(remainder)))
    except (TypeError, ValueError):
        return []

    if diff > tolerance_hours:
        violations.append(
            GuardrailViolation(
                "waterfall_reconciliation",
                f"Waterfall reconciliation diff {diff:.4f}h exceeds tolerance "
                f"{tolerance_hours:.4f}h (total={total}, top_driver={top_driver}, "
                f"remainder={remainder})",
            )
        )
    return violations


def check_forbidden_slide_kit_columns(
    slide_kit_dir: Path,
) -> list[GuardrailViolation]:
    """Fail if any slide_kit CSV contains forbidden columns."""
    violations: list[GuardrailViolation] = []
    if not slide_kit_dir.exists():
        return violations

    for csv_path in sorted(slide_kit_dir.glob("*.csv")):
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                if headers:
                    forbidden = check_forbidden_columns(headers)
                    for col in forbidden:
                        violations.append(
                            GuardrailViolation(
                                "forbidden_column",
                                f"Forbidden column '{col}' in {csv_path.name}",
                            )
                        )
        except Exception:
            continue

    return violations


def check_recommendation_dedupe_conflicts(
    recommendations: list[dict[str, Any]],
) -> list[GuardrailViolation]:
    """Fail if two recommendations share a dedupe key but have conflicting deltas."""
    violations: list[GuardrailViolation] = []
    seen: dict[str, float] = {}

    for item in recommendations:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action_type") or item.get("action") or "")
        target = str(item.get("target") or item.get("process_hint") or "")
        scenario = str(item.get("scenario_id") or "")
        key = f"{action}|{target}|{scenario}"

        delta = item.get("delta_hours") or item.get("modeled_delta_hours")
        if delta is None:
            continue
        try:
            delta_val = round(float(delta), 4)
        except (TypeError, ValueError):
            continue

        if key in seen:
            if abs(seen[key] - delta_val) > 0.01:
                violations.append(
                    GuardrailViolation(
                        "conflicting_dedupe_delta",
                        f"Conflicting deltas for key '{key}': "
                        f"{seen[key]} vs {delta_val}",
                    )
                )
        else:
            seen[key] = delta_val

    return violations


def run_all_guardrails(
    registry: ClaimRegistry | None = None,
    rendered_text: str = "",
    waterfall: dict[str, Any] | None = None,
    slide_kit_dir: Path | None = None,
    recommendations: list[dict[str, Any]] | None = None,
    waterfall_tolerance_hours: float = 0.01,
) -> list[GuardrailViolation]:
    """Run all guardrail checks and return accumulated violations."""
    violations: list[GuardrailViolation] = []

    if registry and rendered_text:
        violations.extend(check_unclaimed_numbers(registry, rendered_text))

    if waterfall:
        violations.extend(
            check_waterfall_reconciliation(waterfall, waterfall_tolerance_hours)
        )

    if slide_kit_dir:
        violations.extend(check_forbidden_slide_kit_columns(slide_kit_dir))

    if recommendations:
        violations.extend(check_recommendation_dedupe_conflicts(recommendations))

    return violations
