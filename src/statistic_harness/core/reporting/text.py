from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..utils import json_dumps
from .config import (
    _read_positive_int_env,
    _REPORT_MD_MAX_FINDINGS_PER_PLUGIN_ENV,
    _REPORT_MD_MAX_STRING_LEN_ENV,
    _REPORT_MD_MAX_EVIDENCE_IDS_ENV,
)
from .matching import _matches_expected


def _trim_long_text(value: Any, *, max_chars: int) -> Any:
    if isinstance(value, str) and len(value) > max_chars:
        suffix = f"... [truncated {len(value) - max_chars} chars]"
        return value[:max_chars] + suffix
    return value


def _trim_finding_for_markdown(
    finding: Any,
    *,
    max_chars: int,
    max_evidence_ids: int,
) -> Any:
    if not isinstance(finding, dict):
        return _trim_long_text(finding, max_chars=max_chars)
    out: dict[str, Any] = {}
    for key, value in finding.items():
        if isinstance(value, str):
            out[key] = _trim_long_text(value, max_chars=max_chars)
            continue
        if key == "evidence" and isinstance(value, dict):
            ev = dict(value)
            for id_key in ("row_ids", "column_ids"):
                ids = ev.get(id_key)
                if isinstance(ids, list) and len(ids) > max_evidence_ids:
                    omitted = len(ids) - max_evidence_ids
                    ev[id_key] = list(ids[:max_evidence_ids]) + [f"... ({omitted} more)"]
            out[key] = ev
            continue
        out[key] = value
    return out


def _trim_plugin_dump_for_markdown(plugin_payload: Any) -> Any:
    if not isinstance(plugin_payload, dict):
        return plugin_payload
    max_findings = _read_positive_int_env(_REPORT_MD_MAX_FINDINGS_PER_PLUGIN_ENV, 120)
    max_chars = _read_positive_int_env(_REPORT_MD_MAX_STRING_LEN_ENV, 500)
    max_evidence_ids = _read_positive_int_env(_REPORT_MD_MAX_EVIDENCE_IDS_ENV, 20)

    out = dict(plugin_payload)
    findings = plugin_payload.get("findings")
    if isinstance(findings, list):
        trimmed_findings = [
            _trim_finding_for_markdown(
                item,
                max_chars=max_chars,
                max_evidence_ids=max_evidence_ids,
            )
            for item in findings[:max_findings]
        ]
        if len(findings) > max_findings:
            trimmed_findings.append(
                {
                    "kind": "report_dump_truncated",
                    "reason": f"trimmed to first {max_findings} findings for report.md",
                    "omitted_findings": len(findings) - max_findings,
                }
            )
        out["findings"] = trimmed_findings
    summary = out.get("summary")
    if isinstance(summary, str):
        out["summary"] = _trim_long_text(summary, max_chars=max_chars)
    return out


def _recommendation_text(status: str, label: str, process_hint: str) -> str:
    suffix = f" (process {process_hint})" if process_hint else ""
    if status == "confirmed":
        return f"Act on {label}{suffix}."
    if status == "over_limit":
        return f"Investigate excess occurrences of {label}{suffix}."
    if status in {"missing", "below_min"}:
        return f"Missing evidence for {label}{suffix}; check inputs and re-run."
    return f"Review {label}{suffix}."


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (int, bool)):
        return str(value)
    if isinstance(value, list):
        preview = ", ".join(str(v) for v in value[:5])
        if len(value) > 5:
            preview += ", ..."
        return f"[{preview}]"
    if isinstance(value, dict):
        return "{...}"
    text = str(value)
    if len(text) > 80:
        return text[:77] + "..."
    return text


def _format_metrics(metrics: dict[str, Any]) -> list[str]:
    if not isinstance(metrics, dict):
        return []
    items: list[tuple[str, Any]] = []
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)):
            items.append((key, value))
    items = sorted(items, key=lambda item: item[0])
    return [f"{key}: {_format_value(value)}" for key, value in items]


def _format_findings(findings: list[Any]) -> list[str]:
    rendered: list[str] = []
    for finding in findings:
        if not isinstance(finding, dict):
            rendered.append(_format_value(finding))
            continue
        kind = finding.get("kind", "finding")
        measurement = finding.get("measurement_type", "measured")
        parts = [f"kind={kind}", f"measurement={measurement}"]
        key_fields = [
            "role",
            "column",
            "process",
            "process_norm",
            "process_id",
            "process_name",
            "module",
            "module_cd",
            "user",
            "user_id",
            "dimension",
            "key",
            "sequence",
            "host",
            "feature",
            "metric",
        ]
        for field in key_fields:
            if field in finding and finding[field] not in (None, ""):
                parts.append(f"{field}={_format_value(finding[field])}")
        numeric_fields = [
            key
            for key, value in finding.items()
            if isinstance(value, (int, float))
            and key not in {"row_index"}
            and (
                key.endswith("_sec")
                or key.endswith("_hours")
                or key.endswith("_count")
                or key.endswith("_runs")
                or key.endswith("_ratio")
                or key.endswith("_pct")
                or key in {"p50", "p95", "p99", "mean", "min", "max", "score"}
            )
        ]
        for key in sorted(numeric_fields)[:6]:
            parts.append(f"{key}={_format_value(finding[key])}")
        evidence = finding.get("evidence")
        if isinstance(evidence, dict):
            row_ids = evidence.get("row_ids")
            col_ids = evidence.get("column_ids")
            if isinstance(row_ids, list) and row_ids:
                parts.append(f"rows={len(row_ids)}")
            if isinstance(col_ids, list) and col_ids:
                parts.append(f"cols={len(col_ids)}")
            query = evidence.get("query")
            if query:
                parts.append(f"query={_format_value(query)}")
        rendered.append(", ".join(parts))
    return rendered


def _format_issue_value(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, int):
            return f"{value}"
        if float(value).is_integer():
            return f"{int(value)}"
        return f"{float(value):.{digits}f}"
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


def _collapse_findings(
    findings: list[Any], max_examples: int = 10
) -> dict[str, Any]:
    if not isinstance(findings, list) or not findings:
        return {"count": 0, "unique_count": 0, "top_examples": []}
    deduped: dict[str, dict[str, Any]] = {}
    for item in findings:
        key = json_dumps(item) if isinstance(item, (dict, list)) else str(item)
        if key not in deduped:
            deduped[key] = {"count": 0, "example": item}
        deduped[key]["count"] += 1
    ordered = sorted(
        deduped.values(), key=lambda entry: entry["count"], reverse=True
    )
    total = len(findings)
    return {
        "count": total,
        "unique_count": len(deduped),
        "top_examples": ordered[:max_examples],
    }


def _plugin_summary_rows(report: dict[str, Any]) -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]]]:
    rows: list[tuple[str, int, str]] = []
    plugins = report.get("plugins", {}) or {}
    for plugin_id, data in plugins.items():
        if not isinstance(data, dict):
            continue
        findings = data.get("findings") or []
        summary = (data.get("summary") or "").strip()
        rows.append((plugin_id, len(findings), summary))
    rows.sort()
    yes_rows = [row for row in rows if row[1] > 0]
    no_rows = [row for row in rows if row[1] == 0]
    return yes_rows, no_rows


def _format_plugin_table(rows: list[tuple[str, int, str]]) -> list[str]:
    lines = ["| Plugin | Findings | One-line summary |", "|---|---:|---|"]
    for plugin_id, count, summary in rows:
        lines.append(f"| `{plugin_id}` | {count} | {summary} |")
    return lines


def _format_known_issue_checks(
    report: dict[str, Any], expected: list[dict[str, Any]]
) -> list[str]:
    items: list[str] = []
    for entry in expected:
        if not isinstance(entry, dict):
            continue
        plugin_id = entry.get("plugin_id")
        kind = entry.get("kind")
        if not kind:
            continue
        where = entry.get("where") or {}
        contains = entry.get("contains") or {}
        min_count = int(entry.get("min_count", 1))
        max_count = entry.get("max_count")
        candidates = []
        for pid, plugin in report.get("plugins", {}).items():
            if plugin_id and pid != plugin_id:
                continue
            for item in plugin.get("findings", []):
                if item.get("kind") == kind:
                    candidates.append(item)
        matches = [item for item in candidates if _matches_expected(item, where, contains)]
        status = "PASS" if len(matches) >= min_count and (max_count is None or len(matches) <= int(max_count)) else "FAIL"
        detail = f"{len(matches)} match(es)"
        title = entry.get("title") or entry.get("description") or ""
        if title:
            title = title.strip()
        context = f"{kind} ({plugin_id or '*'})"
        if title:
            context = f"{title} :: {context}"
        items.append(f"{status} - {context} - {detail}")
    return items
