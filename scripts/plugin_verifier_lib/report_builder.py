"""Report generator for verification results.

Produces verification_report.json and verification_report.md.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .check_runners import VerificationResult
from .issue_classifier import ClassifiedIssue, IssueType, Severity


def build_report(
    results: list[VerificationResult],
    issues: list[ClassifiedIssue],
    output_dir: Path,
    *,
    report_name: str = "verification_report",
    extra_meta: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write JSON and Markdown reports. Returns (json_path, md_path)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build summary stats
    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    errored = sum(1 for r in results if r.status == "ERROR")
    skipped = sum(1 for r in results if r.status == "SKIP")

    severity_counts = {}
    type_counts = {}
    for issue in issues:
        sev = issue.severity.value
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
        itype = issue.issue_type.value
        type_counts[itype] = type_counts.get(itype, 0) + 1

    report_data: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_verifications": total,
            "passed": passed,
            "failed": failed,
            "errored": errored,
            "skipped": skipped,
            "pass_rate": round(passed / total * 100, 1) if total else 0,
            "total_issues": len(issues),
            "severity_counts": severity_counts,
            "type_counts": type_counts,
        },
        "issues": [
            {
                "plugin_id": i.plugin_id,
                "issue_type": i.issue_type.value,
                "severity": i.severity.value,
                "title": i.title,
                "detail": i.detail,
                "failed_checks": i.failed_checks,
                "dataset_name": i.dataset_name,
                "auto_classified": i.auto_classified,
            }
            for i in sorted(issues, key=lambda x: (
                _severity_order(x.severity),
                x.issue_type.value,
                x.plugin_id,
            ))
        ],
        "results": [
            {
                "plugin_id": r.plugin_id,
                "dataset_name": r.dataset_name,
                "status": r.status,
                "plugin_status": r.plugin_status,
                "duration_ms": round(r.duration_ms, 1),
                "checks": [
                    {"name": c.name, "passed": c.passed, "message": c.message}
                    for c in r.check_results
                ],
                "error": r.error,
            }
            for r in results
        ],
    }
    if extra_meta:
        report_data["meta"] = extra_meta

    json_path = output_dir / f"{report_name}.json"
    json_path.write_text(json.dumps(report_data, indent=2, default=str), encoding="utf-8")

    md_path = output_dir / f"{report_name}.md"
    md_path.write_text(_render_markdown(report_data), encoding="utf-8")

    return json_path, md_path


def _severity_order(severity: Severity) -> int:
    return {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(severity.value, 4)


def _render_markdown(data: dict[str, Any]) -> str:
    s = data["summary"]
    lines = [
        "# Plugin Verification Report",
        "",
        f"Generated: {data['generated_at']}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total verifications | {s['total_verifications']} |",
        f"| Passed | {s['passed']} |",
        f"| Failed | {s['failed']} |",
        f"| Errors | {s['errored']} |",
        f"| Skipped | {s['skipped']} |",
        f"| Pass rate | {s['pass_rate']}% |",
        f"| Total issues | {s['total_issues']} |",
        "",
    ]

    if s["severity_counts"]:
        lines.append("### Issues by Severity")
        lines.append("")
        for sev in ("critical", "high", "medium", "low"):
            count = s["severity_counts"].get(sev, 0)
            if count:
                lines.append(f"- **{sev}**: {count}")
        lines.append("")

    if s["type_counts"]:
        lines.append("### Issues by Type")
        lines.append("")
        for itype, count in sorted(s["type_counts"].items()):
            lines.append(f"- `{itype}`: {count}")
        lines.append("")

    # Issues table
    issues = data.get("issues", [])
    if issues:
        lines.append("## Issues")
        lines.append("")
        lines.append("| # | Plugin | Type | Severity | Title |")
        lines.append("|---|---|---|---|---|")
        for idx, issue in enumerate(issues, 1):
            lines.append(
                f"| {idx} | `{issue['plugin_id']}` | "
                f"`{issue['issue_type']}` | {issue['severity']} | "
                f"{issue['title']} |"
            )
        lines.append("")

        # Detailed issue descriptions
        lines.append("### Issue Details")
        lines.append("")
        for idx, issue in enumerate(issues, 1):
            tag = "auto" if issue["auto_classified"] else "pre-classified"
            lines.append(f"**{idx}. {issue['title']}** ({tag})")
            lines.append(f"- Plugin: `{issue['plugin_id']}`")
            lines.append(f"- Type: `{issue['issue_type']}` | Severity: {issue['severity']}")
            if issue["dataset_name"]:
                lines.append(f"- Dataset: `{issue['dataset_name']}`")
            lines.append(f"- Detail: {issue['detail']}")
            if issue["failed_checks"]:
                lines.append(f"- Failed checks: {', '.join(issue['failed_checks'])}")
            lines.append("")

    # Failed results detail
    failed_results = [r for r in data.get("results", []) if r["status"] in ("FAIL", "ERROR")]
    if failed_results:
        lines.append("## Failed Verifications")
        lines.append("")
        for r in failed_results[:50]:  # Limit to 50 for readability
            lines.append(f"### `{r['plugin_id']}` on `{r['dataset_name']}`")
            lines.append(f"- Status: {r['status']} | Plugin status: {r['plugin_status']}")
            lines.append(f"- Duration: {r['duration_ms']:.0f}ms")
            if r["error"]:
                err_lines = r["error"].split("\n")[:5]
                lines.append(f"- Error: `{err_lines[0]}`")
            failed_checks = [c for c in r.get("checks", []) if not c["passed"]]
            for c in failed_checks:
                lines.append(f"  - {c['name']}: {c['message']}")
            lines.append("")

    return "\n".join(lines)
