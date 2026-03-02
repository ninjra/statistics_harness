"""Issue taxonomy classifier for plugin verification failures.

Maps verification check failures to an 8-type issue taxonomy:
  MISLABELED, NUMERICALLY_INCORRECT, ASSUMPTION_VIOLATION,
  DETERMINISM_VIOLATION, MISSING_CORRECTION, MISLEADING_CONFIDENCE,
  SUBSAMPLE_BIAS, CONTRACT_VIOLATION
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .check_runners import CheckResult, VerificationResult


class IssueType(str, Enum):
    MISLABELED = "MISLABELED"
    NUMERICALLY_INCORRECT = "NUMERICALLY_INCORRECT"
    ASSUMPTION_VIOLATION = "ASSUMPTION_VIOLATION"
    DETERMINISM_VIOLATION = "DETERMINISM_VIOLATION"
    MISSING_CORRECTION = "MISSING_CORRECTION"
    MISLEADING_CONFIDENCE = "MISLEADING_CONFIDENCE"
    SUBSAMPLE_BIAS = "SUBSAMPLE_BIAS"
    CONTRACT_VIOLATION = "CONTRACT_VIOLATION"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ClassifiedIssue:
    plugin_id: str
    issue_type: IssueType
    severity: Severity
    title: str
    detail: str
    failed_checks: list[str] = field(default_factory=list)
    dataset_name: str = ""
    auto_classified: bool = True


# Pre-classified issues from code reading (plan Section "Pre-Classified Issues")
PRE_CLASSIFIED: list[dict[str, Any]] = [
    {
        "plugin_id": "analysis_dp_gmm",
        "issue_type": IssueType.MISLABELED,
        "severity": Severity.CRITICAL,
        "title": "Binary mean-threshold, not DP-GMM",
        "detail": "Implementation does a mean threshold, not Dirichlet Process Gaussian Mixture Model. Algorithm replacement needed.",
    },
    {
        "plugin_id": "analysis_bocpd_gaussian",
        "issue_type": IssueType.MISLABELED,
        "severity": Severity.CRITICAL,
        "title": "Cumulative mean deviation, not BOCPD",
        "detail": "Uses cumulative mean deviation, not Bayesian Online Changepoint Detection. Algorithm replacement needed.",
    },
    {
        "plugin_id": "analysis_gaussian_knockoffs",
        "issue_type": IssueType.MISLABELED,
        "severity": Severity.CRITICAL,
        "title": "No knockoff construction; FDR claim invalid",
        "detail": "Missing actual knockoff variable construction. The FDR control claim is invalid without it.",
    },
    {
        "plugin_id": "registry:_two_sample_numeric:ad_branch",
        "issue_type": IssueType.NUMERICALLY_INCORRECT,
        "severity": Severity.HIGH,
        "title": "Fragile p/100 rescaling in AD branch",
        "detail": "p/100 if p>1 is fragile; breaks on modern scipy versions that return correct p-values.",
    },
    {
        "plugin_id": "stats:cliffs_delta",
        "issue_type": IssueType.SUBSAMPLE_BIAS,
        "severity": Severity.MEDIUM,
        "title": "np.linspace subsampling biases sorted data",
        "detail": "Uses np.linspace for index selection, biasing toward endpoints in sorted data. Should use rng.choice.",
    },
    {
        "plugin_id": "leftfield:tensor_cp_parafac",
        "issue_type": IssueType.DETERMINISM_VIOLATION,
        "severity": Severity.MEDIUM,
        "title": "Hardcoded rng(42) ignores ctx.run_seed",
        "detail": "Uses hardcoded seed 42 instead of ctx.run_seed, violating determinism contract.",
    },
    {
        "plugin_id": "analysis_conformal_feature_prediction",
        "issue_type": IssueType.ASSUMPTION_VIOLATION,
        "severity": Severity.HIGH,
        "title": "Sequential split violates exchangeability",
        "detail": "Conformal prediction requires exchangeability; sequential split breaks this assumption.",
    },
    {
        "plugin_id": "leftfield:lingam",
        "issue_type": IssueType.MISLEADING_CONFIDENCE,
        "severity": Severity.MEDIUM,
        "title": "abs(c) discards edge sign",
        "detail": "Taking absolute value of causal coefficients discards directional information.",
    },
    {
        "plugin_id": "analysis_normalizing_flow_density_v1",
        "issue_type": IssueType.MISLABELED,
        "severity": Severity.LOW,
        "title": "Rank-based Gaussianization, not a normalizing flow",
        "detail": "Implementation uses rank-based Gaussianization, not an actual normalizing flow. Works but misnamed.",
    },
    {
        "plugin_id": "registry:_chi2_p_value_fallback",
        "issue_type": IssueType.NUMERICALLY_INCORRECT,
        "severity": Severity.MEDIUM,
        "title": "exp(-0.5*chi2) wrong for df != 2",
        "detail": "Fallback chi-square p-value approximation exp(-0.5*chi2) is only correct for df=2.",
    },
]


def classify_verification_failure(vr: VerificationResult) -> list[ClassifiedIssue]:
    """Classify a failed verification result into issue types."""
    if vr.status in ("PASS", "SKIP"):
        return []

    issues: list[ClassifiedIssue] = []
    failed = [c for c in vr.check_results if not c.passed]

    # Check for known pre-classified issues
    for pre in PRE_CLASSIFIED:
        pid = pre["plugin_id"]
        # Exact match, or plugin_id matches the full colon-delimited prefix
        if pid == vr.plugin_id or (
            ":" in pid and vr.plugin_id == pid.split(":")[0]
        ):
            issues.append(ClassifiedIssue(
                plugin_id=vr.plugin_id,
                issue_type=pre["issue_type"],
                severity=pre["severity"],
                title=pre["title"],
                detail=pre["detail"],
                failed_checks=[c.name for c in failed],
                dataset_name=vr.dataset_name,
                auto_classified=False,
            ))

    if issues:
        return issues

    # Auto-classify based on check failure patterns
    failed_names = {c.name for c in failed}

    if "contract_valid" in failed_names:
        issues.append(ClassifiedIssue(
            plugin_id=vr.plugin_id,
            issue_type=IssueType.CONTRACT_VIOLATION,
            severity=Severity.HIGH,
            title=f"Contract violation in {vr.plugin_id}",
            detail=_get_message(failed, "contract_valid"),
            failed_checks=[c.name for c in failed],
            dataset_name=vr.dataset_name,
        ))

    if "p_values_valid" in failed_names:
        issues.append(ClassifiedIssue(
            plugin_id=vr.plugin_id,
            issue_type=IssueType.NUMERICALLY_INCORRECT,
            severity=Severity.HIGH,
            title=f"Invalid p-values in {vr.plugin_id}",
            detail=_get_message(failed, "p_values_valid"),
            failed_checks=["p_values_valid"],
            dataset_name=vr.dataset_name,
        ))

    if "effect_sizes_finite" in failed_names:
        issues.append(ClassifiedIssue(
            plugin_id=vr.plugin_id,
            issue_type=IssueType.NUMERICALLY_INCORRECT,
            severity=Severity.MEDIUM,
            title=f"Non-finite effect sizes in {vr.plugin_id}",
            detail=_get_message(failed, "effect_sizes_finite"),
            failed_checks=["effect_sizes_finite"],
            dataset_name=vr.dataset_name,
        ))

    if "no_false_findings" in failed_names:
        issues.append(ClassifiedIssue(
            plugin_id=vr.plugin_id,
            issue_type=IssueType.ASSUMPTION_VIOLATION,
            severity=Severity.HIGH,
            title=f"False positives on null data in {vr.plugin_id}",
            detail=_get_message(failed, "no_false_findings"),
            failed_checks=["no_false_findings"],
            dataset_name=vr.dataset_name,
        ))

    if "findings_present" in failed_names:
        issues.append(ClassifiedIssue(
            plugin_id=vr.plugin_id,
            issue_type=IssueType.NUMERICALLY_INCORRECT,
            severity=Severity.MEDIUM,
            title=f"Missing expected findings in {vr.plugin_id}",
            detail=_get_message(failed, "findings_present"),
            failed_checks=["findings_present"],
            dataset_name=vr.dataset_name,
        ))

    if "determinism" in failed_names or "determinism_identical" in failed_names:
        issues.append(ClassifiedIssue(
            plugin_id=vr.plugin_id,
            issue_type=IssueType.DETERMINISM_VIOLATION,
            severity=Severity.MEDIUM,
            title=f"Non-deterministic output from {vr.plugin_id}",
            detail=_get_message(failed, "determinism"),
            failed_checks=["determinism"],
            dataset_name=vr.dataset_name,
        ))

    # Catch-all for unclassified failures
    if not issues and failed:
        issues.append(ClassifiedIssue(
            plugin_id=vr.plugin_id,
            issue_type=IssueType.NUMERICALLY_INCORRECT,
            severity=Severity.MEDIUM,
            title=f"Verification failure in {vr.plugin_id}",
            detail="; ".join(f"{c.name}: {c.message}" for c in failed),
            failed_checks=[c.name for c in failed],
            dataset_name=vr.dataset_name,
        ))

    # ERROR status (exception during run)
    if vr.status == "ERROR" and not issues:
        issues.append(ClassifiedIssue(
            plugin_id=vr.plugin_id,
            issue_type=IssueType.CONTRACT_VIOLATION,
            severity=Severity.HIGH,
            title=f"Runtime error in {vr.plugin_id}",
            detail=vr.error or "Unknown error",
            failed_checks=[],
            dataset_name=vr.dataset_name,
        ))

    return issues


def _get_message(checks: list[CheckResult], name: str) -> str:
    for c in checks:
        if c.name == name:
            return c.message
    return ""
