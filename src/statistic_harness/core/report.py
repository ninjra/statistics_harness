from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import validate

from .dataset_io import DatasetAccessor
from .storage import Storage
from .utils import now_iso, read_json, write_json


def build_report(storage: Storage, run_id: str, run_dir: Path, schema_path: Path) -> dict[str, Any]:
    canonical = run_dir / "dataset" / "canonical.csv"
    accessor = DatasetAccessor(canonical)
    info = accessor.info()

    plugin_rows = storage.fetch_plugin_results(run_id)
    plugins: dict[str, Any] = {}
    for row in plugin_rows:
        plugins[row["plugin_id"]] = {
            "status": row["status"],
            "summary": row["summary"],
            "metrics": json.loads(row["metrics_json"]),
            "findings": json.loads(row["findings_json"]),
            "artifacts": json.loads(row["artifacts_json"]),
            "error": json.loads(row["error_json"]) if row["error_json"] else None,
        }

    report = {
        "run_id": run_id,
        "created_at": now_iso(),
        "status": "completed",
        "input": {
            "filename": canonical.name,
            **info,
        },
        "plugins": plugins,
    }

    schema = read_json(schema_path)
    validate(instance=report, schema=schema)
    return report


def write_report(report: dict[str, Any], run_dir: Path) -> None:
    report_path = run_dir / "report.json"
    write_json(report_path, report)

    lines = ["# Statistic Harness Report", "", "## Dataset", ""]
    lines.append(f"Rows: {report['input']['rows']}")
    lines.append(f"Cols: {report['input']['cols']}")
    lines.append("")
    lines.append("## Plugins")
    for plugin_id, data in report["plugins"].items():
        lines.append(f"- **{plugin_id}** ({data['status']}): {data['summary']}")
    lines.append("")
    lines.append("## Findings")
    for plugin_id, data in report["plugins"].items():
        for finding in data["findings"]:
            lines.append(f"- {plugin_id}: {finding}")
    lines.append("")
    lines.append("## Errors")
    for plugin_id, data in report["plugins"].items():
        if data["error"]:
            lines.append(f"- {plugin_id}: {data['error']['message']}")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
