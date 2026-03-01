"""Verification case/result dataclasses and standard check battery.

Provides the execution engine that runs a plugin against a dataset,
applies verification checks, and records structured results.
"""
from __future__ import annotations

import math
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from statistic_harness.core.stat_plugins.contract import VALID_STATUS, validate_contract
from statistic_harness.core.types import PluginResult


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str


@dataclass
class VerificationCase:
    plugin_id: str
    dataset_name: str
    run_fn: Callable[[], PluginResult]
    checks: list[Callable[[PluginResult, dict[str, Any]], CheckResult]]
    known_answers: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    plugin_id: str
    dataset_name: str
    status: str  # PASS, FAIL, ERROR, SKIP
    check_results: list[CheckResult] = field(default_factory=list)
    error: str | None = None
    duration_ms: float = 0.0
    plugin_status: str | None = None
    raw_metrics: dict[str, Any] = field(default_factory=dict)


def result_to_dict(result: PluginResult, plugin_id: str) -> dict[str, Any]:
    """Convert PluginResult to the dict format expected by validate_contract."""
    return {
        "plugin_id": plugin_id,
        "version": "0.1.0",
        "status": result.status,
        "summary": result.summary,
        "findings": result.findings,
        "artifacts": [
            {"path": a.path, "type": a.type, "description": a.description}
            for a in result.artifacts
        ],
        "metrics": result.metrics,
        "references": result.references,
        "debug": result.debug,
    }


# ---------------------------------------------------------------------------
# Standard check functions
# ---------------------------------------------------------------------------


def check_contract(result: PluginResult, known: dict[str, Any], plugin_id: str = "") -> CheckResult:
    """Validate the result against the plugin contract schema."""
    result_dict = result_to_dict(result, plugin_id or known.get("_plugin_id", "unknown"))
    errors = validate_contract(result_dict)
    if errors:
        return CheckResult("contract_valid", False, f"Contract violations: {errors}")
    return CheckResult("contract_valid", True, "OK")


def check_status_valid(result: PluginResult, known: dict[str, Any]) -> CheckResult:
    """Verify status is in the valid set."""
    if result.status in VALID_STATUS:
        return CheckResult("status_valid", True, f"status={result.status}")
    return CheckResult("status_valid", False, f"Invalid status: {result.status}")


def check_no_error(result: PluginResult, known: dict[str, Any]) -> CheckResult:
    """Verify no uncaught error in result."""
    if result.error is not None:
        return CheckResult("no_error", False, f"Error: {result.error.message}")
    if result.status == "error":
        return CheckResult("no_error", False, f"Status is error: {result.summary}")
    return CheckResult("no_error", True, "OK")


def check_p_values_valid(result: PluginResult, known: dict[str, Any]) -> CheckResult:
    """Verify all p-values in metrics and findings are in [0, 1]."""
    issues = []

    def _check_dict(d: dict, path: str) -> None:
        for k, v in d.items():
            if "p_value" in k.lower() or "pvalue" in k.lower() or k == "p":
                if isinstance(v, (int, float)) and not math.isnan(v):
                    if v < 0 or v > 1:
                        issues.append(f"{path}.{k}={v}")

    _check_dict(result.metrics, "metrics")
    for i, f in enumerate(result.findings):
        _check_dict(f, f"finding[{i}]")
        evidence = f.get("evidence", {})
        if isinstance(evidence, dict):
            _check_dict(evidence, f"finding[{i}].evidence")
            for ek, ev in evidence.items():
                if isinstance(ev, dict):
                    _check_dict(ev, f"finding[{i}].evidence.{ek}")

    if issues:
        return CheckResult("p_values_valid", False, f"Out-of-range p-values: {issues}")
    return CheckResult("p_values_valid", True, "OK")


def check_effect_sizes_finite(result: PluginResult, known: dict[str, Any]) -> CheckResult:
    """Verify effect sizes are finite numbers."""
    issues = []

    def _check_dict(d: dict, path: str) -> None:
        for k, v in d.items():
            if "effect" in k.lower() or "cohens" in k.lower() or "cliff" in k.lower():
                if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
                    issues.append(f"{path}.{k}={v}")

    _check_dict(result.metrics, "metrics")
    for i, f in enumerate(result.findings):
        _check_dict(f, f"finding[{i}]")
        evidence = f.get("evidence", {})
        if isinstance(evidence, dict):
            _check_dict(evidence, f"finding[{i}].evidence")
            for ek, ev in evidence.items():
                if isinstance(ev, dict):
                    _check_dict(ev, f"finding[{i}].evidence.{ek}")

    if issues:
        return CheckResult("effect_sizes_finite", False, f"Non-finite effect sizes: {issues}")
    return CheckResult("effect_sizes_finite", True, "OK")


def check_findings_present(result: PluginResult, known: dict[str, Any]) -> CheckResult:
    """Verify findings are present when the dataset should trigger them."""
    if known.get("has_changepoint") is True or known.get("two_sample_should_reject") is True:
        if result.status in ("ok",) and len(result.findings) == 0:
            return CheckResult("findings_present", False, "Expected findings but got none")
    return CheckResult("findings_present", True, "OK")


def check_no_false_findings(result: PluginResult, known: dict[str, Any]) -> CheckResult:
    """Verify no spurious findings on null datasets."""
    if known.get("has_changepoint") is False or known.get("expected_no_findings_at_p05") is True:
        critical_findings = [
            f for f in result.findings
            if f.get("severity") in ("warn", "critical")
            and f.get("confidence", 0) > 0.95
        ]
        if len(critical_findings) > 3:
            return CheckResult(
                "no_false_findings", False,
                f"Got {len(critical_findings)} high-confidence findings on null data",
            )
    return CheckResult("no_false_findings", True, "OK")


def check_metric_in_range(
    metric_path: str, min_val: float, max_val: float, label: str = "",
) -> Callable[[PluginResult, dict[str, Any]], CheckResult]:
    """Factory: check a specific metric is within [min_val, max_val]."""
    name = label or f"metric_{metric_path}_in_range"

    def _check(result: PluginResult, known: dict[str, Any]) -> CheckResult:
        parts = metric_path.split(".")
        val = result.metrics
        for p in parts:
            if isinstance(val, dict) and p in val:
                val = val[p]
            else:
                return CheckResult(name, True, f"Metric {metric_path} not present (skipped)")
        if not isinstance(val, (int, float)):
            return CheckResult(name, True, f"Metric {metric_path} not numeric (skipped)")
        if min_val <= val <= max_val:
            return CheckResult(name, True, f"{metric_path}={val:.4f} in [{min_val}, {max_val}]")
        return CheckResult(name, False, f"{metric_path}={val:.4f} outside [{min_val}, {max_val}]")

    return _check


# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------

# Default check battery applied to every plugin
DEFAULT_CHECKS: list[Callable[[PluginResult, dict[str, Any]], CheckResult]] = [
    check_status_valid,
    check_no_error,
    check_p_values_valid,
    check_effect_sizes_finite,
    check_findings_present,
    check_no_false_findings,
]


def run_verification(case: VerificationCase) -> VerificationResult:
    """Execute a single verification case and return structured results."""
    t0 = time.perf_counter()
    try:
        result = case.run_fn()
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        return VerificationResult(
            plugin_id=case.plugin_id,
            dataset_name=case.dataset_name,
            status="ERROR",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            duration_ms=elapsed,
        )
    elapsed = (time.perf_counter() - t0) * 1000

    # Inject plugin_id for contract check
    known_with_id = {**case.known_answers, "_plugin_id": case.plugin_id}

    check_results = []
    # Always run contract check first
    cr = check_contract(result, known_with_id, case.plugin_id)
    check_results.append(cr)
    # Run all configured checks
    for check_fn in case.checks:
        try:
            cr = check_fn(result, known_with_id)
            check_results.append(cr)
        except Exception as exc:
            check_results.append(CheckResult(
                getattr(check_fn, "__name__", "unknown_check"),
                False,
                f"Check raised exception: {exc}",
            ))

    all_passed = all(c.passed for c in check_results)
    status = "PASS" if all_passed else "FAIL"

    return VerificationResult(
        plugin_id=case.plugin_id,
        dataset_name=case.dataset_name,
        status=status,
        check_results=check_results,
        duration_ms=elapsed,
        plugin_status=result.status,
        raw_metrics=result.metrics if isinstance(result.metrics, dict) else {},
    )
