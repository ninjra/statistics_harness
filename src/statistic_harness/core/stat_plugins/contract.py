from __future__ import annotations

from typing import Any


REQUIRED_TOP_KEYS = {
    "plugin_id",
    "version",
    "status",
    "summary",
    "findings",
    "artifacts",
    "metrics",
    "references",
    "debug",
}

VALID_STATUS = {"ok", "na", "skipped", "error"}
VALID_SEVERITY = {"info", "warn", "critical"}


def validate_contract(result: dict[str, Any], max_findings: int | None = None) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_TOP_KEYS - set(result.keys())
    if missing:
        errors.append(f"missing_top_keys: {sorted(missing)}")
    status = result.get("status")
    if status not in VALID_STATUS:
        errors.append("invalid_status")
    findings = result.get("findings")
    if not isinstance(findings, list):
        errors.append("findings_not_list")
        return errors
    if max_findings is not None and len(findings) > max_findings:
        errors.append("findings_exceed_max")
    for idx, finding in enumerate(findings):
        if not isinstance(finding, dict):
            errors.append(f"finding_{idx}_not_dict")
            continue
        for key in ("id", "severity", "confidence", "title", "what", "why"):
            if key not in finding:
                errors.append(f"finding_{idx}_missing_{key}")
        severity = finding.get("severity")
        if severity is not None and severity not in VALID_SEVERITY:
            errors.append(f"finding_{idx}_invalid_severity")
    return errors
