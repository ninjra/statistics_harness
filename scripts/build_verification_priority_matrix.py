#!/usr/bin/env python3
"""Build unified verification priority matrix from all verification results.

Merges handler, full-impl, and determinism results, scores each issue,
and produces a prioritized remediation report.

Scoring formula:
    severity_score = (
        algorithmic_impact * 40     # How wrong is output (0-40)
      + user_visibility * 30        # Reaches report.md recommendations? (0-30)
      + plugin_frequency * 20       # Shared primitive? Affects N plugins? (0-20)
      + fix_effort_inverse * 10     # Easy fix = high urgency (0-10)
    )

Usage:
    .venv/bin/python scripts/build_verification_priority_matrix.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plugin_verifier_lib.issue_classifier import (
    ClassifiedIssue,
    IssueType,
    PRE_CLASSIFIED,
    Severity,
)

VERIFICATION_DIR = ROOT / "appdata" / "verification"

# Registry handler count per shared primitive (approximate)
# Used for plugin_frequency scoring
SHARED_HANDLER_COUNTS: dict[str, int] = {
    "registry:_two_sample_numeric": 3,
    "registry:_chi2_p_value_fallback": 5,
    "stats:cliffs_delta": 10,
}


def _load_json(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        print(f"  Warning: {path} not found, skipping")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _algorithmic_impact(issue: dict[str, Any]) -> float:
    """Score 0-40 based on how wrong the output is."""
    itype = issue["issue_type"]
    severity = issue["severity"]
    base = {
        "MISLABELED": 35,
        "NUMERICALLY_INCORRECT": 30,
        "ASSUMPTION_VIOLATION": 28,
        "MISSING_CORRECTION": 32,
        "DETERMINISM_VIOLATION": 15,
        "MISLEADING_CONFIDENCE": 20,
        "SUBSAMPLE_BIAS": 18,
        "CONTRACT_VIOLATION": 25,
    }.get(itype, 15)
    # Severity multiplier
    mult = {"critical": 1.0, "high": 0.85, "medium": 0.6, "low": 0.35}.get(severity, 0.5)
    return min(40, base * mult)


def _user_visibility(issue: dict[str, Any]) -> float:
    """Score 0-30 based on whether incorrect output reaches users."""
    itype = issue["issue_type"]
    # MISLABELED and NUMERICALLY_INCORRECT directly affect report recommendations
    if itype in ("MISLABELED", "NUMERICALLY_INCORRECT", "MISSING_CORRECTION"):
        return 30 if issue["severity"] in ("critical", "high") else 20
    if itype == "ASSUMPTION_VIOLATION":
        return 25
    if itype == "MISLEADING_CONFIDENCE":
        return 20
    if itype == "CONTRACT_VIOLATION":
        return 15
    if itype == "DETERMINISM_VIOLATION":
        return 10
    return 10


def _plugin_frequency(issue: dict[str, Any]) -> float:
    """Score 0-20 based on how many plugins are affected."""
    plugin_id = issue["plugin_id"]
    # Check if it's a shared primitive
    for prefix, count in SHARED_HANDLER_COUNTS.items():
        if plugin_id.startswith(prefix):
            return min(20, count * 4)
    # Single plugin
    return 5


def _fix_effort_inverse(issue: dict[str, Any]) -> float:
    """Score 0-10: easy fix = high urgency (fix it first)."""
    itype = issue["issue_type"]
    # Easy fixes: subsample bias (change rng), determinism violation (propagate seed)
    if itype in ("SUBSAMPLE_BIAS", "DETERMINISM_VIOLATION"):
        return 9
    if itype == "NUMERICALLY_INCORRECT":
        return 7
    if itype == "MISLEADING_CONFIDENCE":
        return 7
    if itype == "CONTRACT_VIOLATION":
        return 6
    # Hard fixes: algorithm replacement for mislabeled
    if itype == "MISLABELED":
        return 3
    if itype == "MISSING_CORRECTION":
        return 3
    if itype == "ASSUMPTION_VIOLATION":
        return 4
    return 5


def _compute_priority_score(issue: dict[str, Any]) -> float:
    return (
        _algorithmic_impact(issue)
        + _user_visibility(issue)
        + _plugin_frequency(issue)
        + _fix_effort_inverse(issue)
    )


def _dedup_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate issues by (plugin_id, issue_type)."""
    seen: set[tuple[str, str]] = set()
    deduped = []
    for issue in issues:
        key = (issue["plugin_id"], issue["issue_type"])
        if key not in seen:
            seen.add(key)
            deduped.append(issue)
    return deduped


def main() -> None:
    print("Building verification priority matrix...")

    all_issues: list[dict[str, Any]] = []

    # Load pre-classified issues
    for pre in PRE_CLASSIFIED:
        all_issues.append({
            "plugin_id": pre["plugin_id"],
            "issue_type": pre["issue_type"].value,
            "severity": pre["severity"].value,
            "title": pre["title"],
            "detail": pre["detail"],
            "source": "pre_classified",
            "auto_classified": False,
        })

    # Load handler verification report
    handler_report = _load_json(VERIFICATION_DIR / "handler_verification_report.json")
    if handler_report and "issues" in handler_report:
        for issue in handler_report["issues"]:
            issue["source"] = "handler_verification"
            all_issues.append(issue)

    # Load full-impl verification report
    full_impl_report = _load_json(VERIFICATION_DIR / "full_impl_verification_report.json")
    if full_impl_report and "issues" in full_impl_report:
        for issue in full_impl_report["issues"]:
            issue["source"] = "full_impl_verification"
            all_issues.append(issue)

    # Load determinism verification report
    det_report = _load_json(VERIFICATION_DIR / "determinism_verification_report.json")
    if det_report and "issues" in det_report:
        for issue in det_report["issues"]:
            issue["source"] = "determinism_verification"
            all_issues.append(issue)

    # Deduplicate
    all_issues = _dedup_issues(all_issues)

    # Score and sort
    scored_issues = []
    for issue in all_issues:
        score = _compute_priority_score(issue)
        issue["priority_score"] = round(score, 1)
        issue["score_breakdown"] = {
            "algorithmic_impact": round(_algorithmic_impact(issue), 1),
            "user_visibility": round(_user_visibility(issue), 1),
            "plugin_frequency": round(_plugin_frequency(issue), 1),
            "fix_effort_inverse": round(_fix_effort_inverse(issue), 1),
        }
        scored_issues.append(issue)

    scored_issues.sort(key=lambda x: x["priority_score"], reverse=True)

    # Impact classification
    system_time_issues = [
        i for i in scored_issues
        if i["issue_type"] in ("NUMERICALLY_INCORRECT", "CONTRACT_VIOLATION", "MISLABELED")
    ]
    human_time_issues = [
        i for i in scored_issues
        if i["issue_type"] in ("MISLABELED", "ASSUMPTION_VIOLATION", "MISLEADING_CONFIDENCE", "MISSING_CORRECTION")
    ]

    # Load summary stats from sub-reports
    handler_summary = {}
    if handler_report and "summary" in handler_report:
        handler_summary = handler_report["summary"]
    full_impl_summary = {}
    if full_impl_report and "summary" in full_impl_report:
        full_impl_summary = full_impl_report["summary"]
    det_summary = {}
    if det_report and "summary" in det_report:
        det_summary = det_report["summary"]

    # Build final report
    report = {
        "summary": {
            "total_issues": len(scored_issues),
            "by_severity": _count_by(scored_issues, "severity"),
            "by_type": _count_by(scored_issues, "issue_type"),
            "by_source": _count_by(scored_issues, "source"),
            "system_time_impact_count": len(system_time_issues),
            "human_time_impact_count": len(human_time_issues),
            "sub_reports": {
                "handler_verification": handler_summary,
                "full_impl_verification": full_impl_summary,
                "determinism_verification": det_summary,
            },
        },
        "priority_matrix": scored_issues,
        "impact_classification": {
            "system_time": [
                {"plugin_id": i["plugin_id"], "issue_type": i["issue_type"],
                 "title": i["title"], "score": i["priority_score"]}
                for i in system_time_issues[:20]
            ],
            "human_time": [
                {"plugin_id": i["plugin_id"], "issue_type": i["issue_type"],
                 "title": i["title"], "score": i["priority_score"]}
                for i in human_time_issues[:20]
            ],
        },
    }

    # Write JSON
    json_path = VERIFICATION_DIR / "verification_report.json"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Write Markdown
    md_path = VERIFICATION_DIR / "verification_report.md"
    md_path.write_text(_render_markdown(report, scored_issues), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Priority Matrix Complete")
    print(f"  Total issues: {len(scored_issues)}")
    print(f"  By severity: {_count_by(scored_issues, 'severity')}")
    print(f"  By type: {_count_by(scored_issues, 'issue_type')}")
    print(f"  System time impact: {len(system_time_issues)}")
    print(f"  Human time impact: {len(human_time_issues)}")
    print(f"  Report: {json_path}")
    print(f"  Report: {md_path}")


def _count_by(items: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        val = item.get(key, "unknown")
        counts[val] = counts.get(val, 0) + 1
    return counts


def _render_markdown(report: dict, issues: list[dict]) -> str:
    s = report["summary"]
    lines = [
        "# Plugin Verification Priority Matrix",
        "",
        "## Summary",
        "",
        f"- **Total issues**: {s['total_issues']}",
        f"- **System time impact**: {s['system_time_impact_count']} issues",
        f"- **Human time impact**: {s['human_time_impact_count']} issues",
        "",
        "### By Severity",
        "",
    ]
    for sev in ("critical", "high", "medium", "low"):
        count = s["by_severity"].get(sev, 0)
        if count:
            lines.append(f"- **{sev}**: {count}")
    lines.append("")
    lines.append("### By Issue Type")
    lines.append("")
    for itype, count in sorted(s["by_type"].items()):
        lines.append(f"- `{itype}`: {count}")
    lines.append("")

    # Sub-report summaries
    for name, sub in s.get("sub_reports", {}).items():
        if sub:
            passed = sub.get("passed", "?")
            total = sub.get("total_verifications", "?")
            lines.append(f"- **{name}**: {passed}/{total} passed")
    lines.append("")

    # Priority matrix table
    lines.append("## Priority Matrix (Top 30)")
    lines.append("")
    lines.append("| Rank | Score | Plugin | Type | Severity | Title |")
    lines.append("|---|---|---|---|---|---|")
    for rank, issue in enumerate(issues[:30], 1):
        lines.append(
            f"| {rank} | {issue['priority_score']} | "
            f"`{issue['plugin_id']}` | `{issue['issue_type']}` | "
            f"{issue['severity']} | {issue['title']} |"
        )
    lines.append("")

    # Detailed breakdown for top 10
    lines.append("## Top 10 Issue Details")
    lines.append("")
    for rank, issue in enumerate(issues[:10], 1):
        lines.append(f"### {rank}. {issue['title']}")
        lines.append(f"- **Plugin**: `{issue['plugin_id']}`")
        lines.append(f"- **Type**: `{issue['issue_type']}` | **Severity**: {issue['severity']}")
        lines.append(f"- **Priority score**: {issue['priority_score']}")
        bd = issue.get("score_breakdown", {})
        lines.append(
            f"  - Algorithmic impact: {bd.get('algorithmic_impact', '?')} / 40"
        )
        lines.append(
            f"  - User visibility: {bd.get('user_visibility', '?')} / 30"
        )
        lines.append(
            f"  - Plugin frequency: {bd.get('plugin_frequency', '?')} / 20"
        )
        lines.append(
            f"  - Fix effort (inverse): {bd.get('fix_effort_inverse', '?')} / 10"
        )
        lines.append(f"- **Detail**: {issue.get('detail', '')}")
        tag = "pre-classified" if not issue.get("auto_classified", True) else "auto"
        lines.append(f"- **Source**: {issue.get('source', '?')} ({tag})")
        lines.append("")

    # Impact classification
    lines.append("## Impact Classification")
    lines.append("")
    lines.append("### System Time Impact")
    lines.append("Incorrect `ok` status forces downstream processing of invalid findings.")
    lines.append("")
    impact = report.get("impact_classification", {})
    for item in impact.get("system_time", [])[:10]:
        lines.append(f"- `{item['plugin_id']}`: {item['title']} (score: {item['score']})")
    lines.append("")
    lines.append("### Human Time Impact")
    lines.append("False recommendations require analyst investigation.")
    lines.append("")
    for item in impact.get("human_time", [])[:10]:
        lines.append(f"- `{item['plugin_id']}`: {item['title']} (score: {item['score']})")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
