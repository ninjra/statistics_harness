from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import validate

from .storage import Storage
from .utils import json_dumps, now_iso, read_json, write_json


def build_report(
    storage: Storage, run_id: str, run_dir: Path, schema_path: Path
) -> dict[str, Any]:
    run_row = storage.fetch_run(run_id)
    if not run_row or not run_row.get("dataset_version_id"):
        raise ValueError("Run dataset version not found")
    upload_row = (
        storage.fetch_upload(run_row["upload_id"])
        if run_row.get("upload_id")
        else None
    )
    from .dataset_io import DatasetAccessor

    accessor = DatasetAccessor(storage, run_row["dataset_version_id"])
    info = accessor.info()

    plugin_rows = storage.fetch_plugin_results(run_id)
    plugins: dict[str, Any] = {}

    def _ensure_measurement(findings: list[Any]) -> list[Any]:
        for item in findings:
            if isinstance(item, dict) and "measurement_type" not in item:
                item["measurement_type"] = "measured"
        return findings

    def _canonicalize_payload(payload: Any) -> Any:
        try:
            return json.loads(json_dumps(payload))
        except TypeError:
            return payload

    def _sort_payload_list(items: list[Any]) -> list[Any]:
        try:
            return sorted(items, key=lambda item: json_dumps(item))
        except Exception:
            return items

    for row in sorted(plugin_rows, key=lambda item: item["plugin_id"]):
        findings = json.loads(row["findings_json"])
        if isinstance(findings, list):
            findings = _ensure_measurement(findings)
            findings = _sort_payload_list(findings)
        artifacts = json.loads(row["artifacts_json"])
        if isinstance(artifacts, list):
            artifacts = _sort_payload_list(artifacts)
        budget = None
        if "budget_json" in row.keys() and row.get("budget_json"):
            try:
                budget = json.loads(row["budget_json"])
            except json.JSONDecodeError:
                budget = None
        if not isinstance(budget, dict):
            budget = {
                "row_limit": None,
                "sampled": False,
                "time_limit_ms": None,
                "cpu_limit_ms": None,
            }
        plugins[row["plugin_id"]] = {
            "status": row["status"],
            "summary": row["summary"],
            "metrics": _canonicalize_payload(json.loads(row["metrics_json"])),
            "findings": findings,
            "artifacts": artifacts,
            "budget": _canonicalize_payload(budget),
            "error": json.loads(row["error_json"]) if row["error_json"] else None,
        }

    dataset_version = storage.get_dataset_version(run_row["dataset_version_id"])
    dataset_context = storage.get_dataset_version_context(run_row["dataset_version_id"])
    project_row = None
    if dataset_context and dataset_context.get("project_id"):
        project_row = storage.fetch_project(dataset_context["project_id"])
    dataset_template = storage.fetch_dataset_template(run_row["dataset_version_id"])
    raw_format = None
    raw_format_id = None
    if dataset_version:
        raw_format_id = dataset_version.get("raw_format_id")
    if raw_format_id:
        with storage.connection() as conn:
            cur = conn.execute(
                """
                SELECT format_id, fingerprint, name, created_at
                FROM raw_formats
                WHERE format_id = ?
                """,
                (raw_format_id,),
            )
            row = cur.fetchone()
            raw_format = dict(row) if row else None
    if raw_format:
        raw_format = {
            "format_id": int(raw_format.get("format_id") or raw_format_id or 0),
            "fingerprint": raw_format.get("fingerprint") or "",
            "name": raw_format.get("name") or "",
            "created_at": raw_format.get("created_at") or "",
        }

    mapping = None
    if dataset_template and dataset_template.get("mapping_json"):
        try:
            mapping = json.loads(dataset_template["mapping_json"])
        except json.JSONDecodeError:
            mapping = {}

    dataset_block = {
        "dataset_version_id": run_row.get("dataset_version_id") or "unknown",
    }
    if dataset_context:
        if dataset_context.get("project_id"):
            dataset_block["project_id"] = dataset_context["project_id"]
        if dataset_context.get("dataset_id"):
            dataset_block["dataset_id"] = dataset_context["dataset_id"]
        if dataset_context.get("table_name"):
            dataset_block["table_name"] = dataset_context["table_name"]
    if dataset_version:
        if dataset_version.get("data_hash"):
            dataset_block["data_hash"] = dataset_version["data_hash"]
        if dataset_version.get("row_count") is not None:
            dataset_block["row_count"] = int(dataset_version["row_count"])
        if dataset_version.get("column_count") is not None:
            dataset_block["column_count"] = int(dataset_version["column_count"])
        if dataset_version.get("raw_format_id"):
            dataset_block["raw_format_id"] = int(dataset_version["raw_format_id"])

    def _string_or_empty(value: Any) -> str:
        return value if isinstance(value, str) else ""

    lineage_plugins: dict[str, Any] = {}
    for row in plugin_rows:
        lineage_plugins[row["plugin_id"]] = {
            "plugin_version": _string_or_empty(row.get("plugin_version")),
            "code_hash": _string_or_empty(row.get("code_hash")),
            "settings_hash": _string_or_empty(row.get("settings_hash")),
            "dataset_hash": _string_or_empty(row.get("dataset_hash")),
            "executed_at": _string_or_empty(row.get("executed_at")),
            "status": _string_or_empty(row.get("status")),
            "summary": _string_or_empty(row.get("summary")),
        }

    template_block = None
    if dataset_template:
        template_block = {
            "template_id": int(dataset_template["template_id"]),
            "table_name": dataset_template.get("table_name") or "",
            "status": dataset_template.get("status") or "",
            "mapping_hash": dataset_template.get("mapping_hash") or "",
            "mapping": mapping if isinstance(mapping, dict) else {},
        }
        if dataset_template.get("template_name"):
            template_block["template_name"] = dataset_template["template_name"]
        if dataset_template.get("template_version"):
            template_block["template_version"] = dataset_template["template_version"]

    known_block = None
    known_scope_type = ""
    known_scope_value = ""
    if project_row and project_row.get("erp_type"):
        known_scope_type = "erp_type"
        known_scope_value = str(project_row.get("erp_type") or "unknown").strip() or "unknown"
        known_block = storage.fetch_known_issues(known_scope_value, known_scope_type)
    if not known_block and upload_row and upload_row.get("sha256"):
        known_scope_type = "sha256"
        known_scope_value = str(upload_row.get("sha256") or "")
        if known_scope_value:
            known_block = storage.fetch_known_issues(known_scope_value, known_scope_type)

    known_payload = None
    if known_block:
        known_payload = {
            "scope_type": known_block.get("scope_type") or known_scope_type,
            "scope_value": known_block.get("scope_value") or known_scope_value,
            "strict": bool(known_block.get("strict", True)),
            "notes": known_block.get("notes") or "",
            "natural_language": known_block.get("natural_language") or [],
            "expected_findings": known_block.get("expected_findings") or [],
        }

    report = {
        "run_id": run_id,
        "created_at": now_iso(),
        "status": "completed",
        "input": {
            "filename": run_row.get("input_filename") or "unknown",
            **info,
        },
        "lineage": {
            "run": {
                "run_id": run_id,
                "created_at": run_row.get("created_at") or "",
                "status": run_row.get("status") or "",
                "run_seed": int(run_row.get("run_seed") or 0),
            },
            "input": {
                "upload_id": run_row.get("upload_id") or "",
                "filename": run_row.get("input_filename") or "unknown",
                "canonical_path": run_row.get("canonical_path") or "",
                "input_hash": run_row.get("input_hash") or "",
                "sha256": upload_row.get("sha256") if upload_row else "",
                "size_bytes": int(upload_row.get("size_bytes") or 0)
                if upload_row
                else 0,
            },
            "dataset": dataset_block,
            "raw_format": raw_format,
            "template": template_block,
            "plugins": lineage_plugins,
        },
        "plugins": plugins,
    }
    if known_payload:
        report["known_issues"] = known_payload
    evaluation_path = run_dir / "evaluation.json"
    if evaluation_path.exists():
        try:
            report["evaluation"] = json.loads(evaluation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report["evaluation"] = None

    schema = read_json(schema_path)
    validate(instance=report, schema=schema)
    return report


def write_report(report: dict[str, Any], run_dir: Path) -> None:
    report_path = run_dir / "report.json"
    write_json(report_path, report)

    lines = ["# Statistic Harness Report", ""]
    known = report.get("known_issues")
    if isinstance(known, dict):
        lines.append("## Known Issues")
        scope_type = known.get("scope_type") or ""
        scope_value = known.get("scope_value") or ""
        if scope_type and scope_value:
            lines.append(f"Scope: {scope_type}={scope_value}")
        natural = known.get("natural_language") or []
        if natural:
            lines.append("")
            lines.append("Declared:")
            for entry in natural:
                text = entry.get("text") if isinstance(entry, dict) else None
                if text:
                    lines.append(f"- {text}")
        expected = known.get("expected_findings") or []
        if expected:
            lines.append("")
            lines.append("Checks:")
            for item in _format_known_issue_checks(report, expected):
                lines.append(f"- {item}")
        lines.append("")

    lines.append("## Dataset")
    lines.append("")
    lines.append(f"Rows: {report['input']['rows']}")
    lines.append(f"Cols: {report['input']['cols']}")
    lineage = report.get("lineage") or {}
    dataset_lineage = lineage.get("dataset") or {}
    run_lineage = lineage.get("run") or {}
    template_lineage = lineage.get("template")
    raw_format_lineage = lineage.get("raw_format")
    lines.append("")
    lines.append("## Lineage")
    lines.append(f"Run Seed: {run_lineage.get('run_seed', 0)}")
    lines.append(f"Dataset Version: {dataset_lineage.get('dataset_version_id', 'unknown')}")
    if dataset_lineage.get("data_hash"):
        lines.append(f"Data Hash: {dataset_lineage['data_hash']}")
    if raw_format_lineage and isinstance(raw_format_lineage, dict):
        lines.append(
            f"Raw Format: {raw_format_lineage.get('fingerprint', '')}"
        )
    if template_lineage and isinstance(template_lineage, dict):
        lines.append(
            f"Template: {template_lineage.get('template_name', '')} "
            f"({template_lineage.get('template_version', '')})"
        )
    lines.append("")
    lines.append("## Plugins")
    for plugin_id, data in report["plugins"].items():
        lines.append(f"### {plugin_id} ({data['status']})")
        if data.get("summary"):
            lines.append(f"Summary: {data['summary']}")
        metrics = data.get("metrics") or {}
        metric_lines = _format_metrics(metrics)
        if metric_lines:
            lines.append("Metrics:")
            for item in metric_lines:
                lines.append(f"- {item}")
        findings = data.get("findings") or []
        if findings:
            lines.append("Findings:")
            for item in _format_findings(findings):
                lines.append(f"- {item}")
        artifacts = data.get("artifacts") or []
        if artifacts:
            lines.append("Artifacts:")
            for artifact in artifacts:
                path = artifact.get("path")
                desc = artifact.get("description") or ""
                if path:
                    lines.append(f"- {path} {('- ' + desc) if desc else ''}".strip())
        if data.get("error"):
            lines.append(f"Error: {data['error'].get('message')}")
        lines.append("")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


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


def _matches_expected(
    item: dict[str, Any], where: dict[str, Any] | None, contains: dict[str, Any] | None
) -> bool:
    if where:
        for key, expected in where.items():
            if item.get(key) != expected:
                return False
    if contains:
        for key, expected in contains.items():
            actual = item.get(key)
            if isinstance(actual, str):
                if str(expected) not in actual:
                    return False
            elif isinstance(actual, (list, tuple, set)):
                if isinstance(expected, (list, tuple, set)):
                    if not set(expected).issubset(set(actual)):
                        return False
                else:
                    if expected not in actual:
                        return False
            else:
                return False
    return True


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
