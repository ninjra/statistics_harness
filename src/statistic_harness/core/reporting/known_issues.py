from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .matching import _matches_expected
from .process_targeting import _process_hint


def _collect_findings_for_plugin(
    report: dict[str, Any], plugin_id: str | None, kind: str | None = None
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    plugins = report.get("plugins", {}) or {}
    for pid, plugin in plugins.items():
        if plugin_id and pid != plugin_id:
            continue
        for item in plugin.get("findings", []) or []:
            if kind and item.get("kind") != kind:
                continue
            if isinstance(item, dict):
                findings.append(item)
    return findings


def _item_process_norm(item: dict[str, Any]) -> str:
    where = item.get("where") if isinstance(item.get("where"), dict) else {}
    for key in ("process_norm", "process", "process_id", "activity"):
        value = where.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    target = item.get("target")
    if isinstance(target, str) and target.strip():
        return target.strip().lower()
    return ""


def _known_recommendation_match(
    report: dict[str, Any], plugin_id: str | None, kind: str | None, process_hint: str
) -> dict[str, Any] | None:
    recommendations = report.get("recommendations")
    if not isinstance(recommendations, dict):
        return None
    known_block = recommendations.get("known")
    if not isinstance(known_block, dict):
        return None
    items = known_block.get("items")
    if not isinstance(items, list):
        return None
    proc = str(process_hint or "").strip().lower()
    for item in items:
        if not isinstance(item, dict):
            continue
        if plugin_id and str(item.get("plugin_id") or "").strip() != str(plugin_id).strip():
            continue
        if kind and str(item.get("kind") or "").strip() != str(kind).strip():
            continue
        if proc and _item_process_norm(item) != proc:
            continue
        return item
    return None


def _known_issue_processes(known: dict[str, Any] | None) -> set[str]:
    processes: set[str] = set()
    if not isinstance(known, dict):
        return processes
    expected = known.get("expected_findings") or []
    if not isinstance(expected, list):
        return processes
    keys = {"process", "process_norm", "process_name", "process_id", "activity", "process_matches"}
    for issue in expected:
        if not isinstance(issue, dict):
            continue
        for bucket in (issue.get("where"), issue.get("contains")):
            if not isinstance(bucket, dict):
                continue
            for key in keys:
                value = bucket.get(key)
                if isinstance(value, str) and value.strip():
                    processes.add(value.strip().lower())
                elif isinstance(value, (list, tuple, set)):
                    for entry in value:
                        if isinstance(entry, str) and entry.strip():
                            processes.add(entry.strip().lower())
    return processes


def _sanitize_known_recommendation_exclusions(known: dict[str, Any] | None) -> None:
    if not isinstance(known, dict):
        return
    required = _known_issue_processes(known)
    if not required:
        return
    exclusions = known.get("recommendation_exclusions")
    if not isinstance(exclusions, dict):
        return
    processes = exclusions.get("processes")
    if not isinstance(processes, list):
        return
    filtered: list[str] = []
    for entry in processes:
        token = str(entry or "").strip()
        if not token:
            continue
        if token.lower() in required:
            continue
        filtered.append(token)
    exclusions["processes"] = sorted(set(filtered))
    known["recommendation_exclusions"] = exclusions


def _evaluate_known_issues(report: dict[str, Any]) -> list[dict[str, Any]]:
    known = report.get("known_issues")
    if not isinstance(known, dict):
        return []
    expected = known.get("expected_findings") or []
    if not isinstance(expected, list) or not expected:
        return []
    evaluations: list[dict[str, Any]] = []
    for issue in expected:
        if not isinstance(issue, dict):
            continue
        plugin_id = issue.get("plugin_id")
        kind = issue.get("kind")
        where = issue.get("where") if isinstance(issue.get("where"), dict) else None
        contains = issue.get("contains") if isinstance(issue.get("contains"), dict) else None
        min_count = issue.get("min_count")
        max_count = issue.get("max_count")
        title = issue.get("title") or issue.get("description") or ""
        label = title or f"{plugin_id or 'any'}:{kind or 'finding'}"

        findings = _collect_findings_for_plugin(report, plugin_id, kind)
        matched = [f for f in findings if _matches_expected(f, where, contains)]
        count = len(matched)
        process_hint = _process_hint(where)
        if count == 0:
            known_item = _known_recommendation_match(
                report, plugin_id, kind, process_hint
            )
            if isinstance(known_item, dict):
                matched = [known_item]
                observed = known_item.get("observed_count")
                if isinstance(observed, (int, float)):
                    count = int(observed)
                else:
                    count = 1

        status = "confirmed"
        if count == 0:
            status = "missing"
        if min_count is not None:
            try:
                if count < int(min_count):
                    status = "below_min"
            except (TypeError, ValueError):
                pass
        if max_count is not None:
            try:
                if count > int(max_count):
                    status = "over_limit"
            except (TypeError, ValueError):
                pass

        evaluations.append(
            {
                "issue": issue,
                "label": label,
                "plugin_id": plugin_id,
                "kind": kind,
                "where": where,
                "contains": contains,
                "min_count": min_count,
                "max_count": max_count,
                "matched": matched,
                "count": count,
                "status": status,
                "process_hint": process_hint,
            }
        )
    return evaluations


def _load_known_issues_fallback(run_dir: Path) -> dict[str, Any] | None:
    known_dir = run_dir.parent.parent / "known_issues"
    if not known_dir.exists():
        return None
    payloads: list[dict[str, Any]] = []
    for path in sorted(known_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("expected_findings"):
            payloads.append(data)
    if not payloads:
        return None
    expected: list[dict[str, Any]] = []
    strict_values: list[bool] = []
    notes: list[str] = []
    for data in payloads:
        strict_values.append(bool(data.get("strict", False)))
        note = str(data.get("notes") or "").strip()
        if note:
            notes.append(note)
        expected.extend(data.get("expected_findings") or [])
    note_text = "Fallback merged from appdata/known_issues"
    if notes:
        note_text = f"{note_text} | " + " | ".join(notes)
    return {
        "scope_type": "fallback",
        "scope_value": "appdata/known_issues",
        "strict": all(strict_values) if strict_values else False,
        "notes": note_text,
        "natural_language": [],
        "expected_findings": expected,
    }
