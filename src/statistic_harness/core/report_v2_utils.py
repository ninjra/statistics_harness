from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .dataset_io import DatasetAccessor
from .storage import Storage
from .utils import file_sha256, json_dumps, stable_hash


def load_plugin_payloads(storage: Storage, run_id: str) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for row in storage.fetch_plugin_results(run_id):
        try:
            findings = json.loads(row["findings_json"])
        except json.JSONDecodeError:
            findings = []
        try:
            artifacts = json.loads(row["artifacts_json"])
        except json.JSONDecodeError:
            artifacts = []
        try:
            metrics = json.loads(row["metrics_json"])
        except json.JSONDecodeError:
            metrics = {}
        payloads[row["plugin_id"]] = {
            "status": row.get("status"),
            "summary": row.get("summary"),
            "metrics": metrics if isinstance(metrics, dict) else {},
            "findings": findings if isinstance(findings, list) else [],
            "artifacts": artifacts if isinstance(artifacts, list) else [],
        }
    return payloads


def _load_known_issues_fallback(run_dir: Path) -> dict[str, Any] | None:
    from .report import _load_known_issues_fallback as _fallback

    return _fallback(run_dir)


def _normalize_known_issues(
    known_block: dict[str, Any], scope_type: str, scope_value: str
) -> dict[str, Any]:
    payload = {
        "scope_type": known_block.get("scope_type") or scope_type,
        "scope_value": known_block.get("scope_value") or scope_value,
        "strict": bool(known_block.get("strict", True)),
        "notes": known_block.get("notes") or "",
        "natural_language": known_block.get("natural_language") or [],
        "expected_findings": known_block.get("expected_findings") or [],
    }
    return payload


def _apply_quorum_exclusions(
    known_payload: dict[str, Any] | None, project_row: dict[str, Any] | None
) -> dict[str, Any] | None:
    if not project_row:
        return known_payload
    if str(project_row.get("erp_type") or "").strip().lower() != "quorum":
        return known_payload
    if not known_payload:
        known_payload = {
            "scope_type": "erp_type",
            "scope_value": "quorum",
            "strict": False,
            "notes": "",
            "natural_language": [],
            "expected_findings": [],
        }
    exclusions = known_payload.get("recommendation_exclusions")
    if not isinstance(exclusions, dict):
        exclusions = {}
    processes = exclusions.get("processes")
    if not isinstance(processes, list):
        processes = []
    quorum_exclusions = {"postwkfl", "bkrvnu", "cwowfndrls", "los", "qemail", "qpec"}
    merged = sorted({*(str(p).strip() for p in processes if str(p).strip()), *quorum_exclusions})
    exclusions["processes"] = merged
    known_payload["recommendation_exclusions"] = exclusions
    return known_payload


def load_known_issues(
    storage: Storage,
    run_id: str | None,
    run_dir: Path,
    project_id: str | None = None,
    dataset_version_id: str | None = None,
) -> dict[str, Any] | None:
    run_row = storage.fetch_run(run_id) if run_id else None
    project_row = storage.fetch_project(project_id) if project_id else None
    if not project_row and run_row and run_row.get("project_id"):
        project_row = storage.fetch_project(run_row.get("project_id"))

    known_block = None
    known_scope_type = ""
    known_scope_value = ""
    if project_row and project_row.get("erp_type"):
        known_scope_type = "erp_type"
        known_scope_value = str(project_row.get("erp_type") or "unknown").strip() or "unknown"
        known_block = storage.fetch_known_issues(known_scope_value, known_scope_type)

    upload_row = None
    if not known_block and run_row and run_row.get("upload_id"):
        upload_row = storage.fetch_upload(run_row.get("upload_id"))
        if upload_row and upload_row.get("sha256"):
            known_scope_type = "sha256"
            known_scope_value = str(upload_row.get("sha256") or "")
            if known_scope_value:
                known_block = storage.fetch_known_issues(known_scope_value, known_scope_type)

    if not known_block:
        if dataset_version_id is None and run_row:
            dataset_version_id = run_row.get("dataset_version_id")
        if dataset_version_id:
            version = storage.get_dataset_version(dataset_version_id)
            data_hash = None
            if version and version.get("data_hash"):
                data_hash = str(version.get("data_hash") or "")
            if data_hash and re.fullmatch(r"[a-f0-9]{64}", data_hash):
                known_scope_type = "sha256"
                known_scope_value = data_hash
                known_block = storage.fetch_known_issues(known_scope_value, known_scope_type)

    known_payload = None
    if known_block:
        known_payload = _normalize_known_issues(
            known_block, known_scope_type, known_scope_value
        )
    if not known_payload:
        known_payload = _load_known_issues_fallback(run_dir)

    known_payload = _apply_quorum_exclusions(known_payload, project_row)
    return known_payload


def build_minimal_report(
    storage: Storage,
    run_id: str,
    run_dir: Path,
    project_id: str | None = None,
    dataset_version_id: str | None = None,
) -> dict[str, Any]:
    run_row = storage.fetch_run(run_id)
    if dataset_version_id is None and run_row:
        dataset_version_id = run_row.get("dataset_version_id")
    info = {"rows": 0, "cols": 0, "inferred_types": {}}
    if dataset_version_id:
        try:
            accessor = DatasetAccessor(storage, dataset_version_id)
            info = accessor.info()
        except Exception:
            pass
    return {
        "run_id": run_id,
        "input": info,
        "plugins": load_plugin_payloads(storage, run_id),
        "known_issues": load_known_issues(
            storage,
            run_id,
            run_dir,
            project_id=project_id,
            dataset_version_id=dataset_version_id,
        ),
    }


def artifact_paths_for_plugin(
    plugins: dict[str, Any], plugin_id: str | None
) -> list[str]:
    if not plugin_id:
        return []
    plugin = plugins.get(plugin_id)
    if not isinstance(plugin, dict):
        return []
    artifacts = plugin.get("artifacts") or []
    paths: list[str] = []
    for artifact in artifacts:
        if isinstance(artifact, dict):
            path = artifact.get("path")
            if isinstance(path, str) and path:
                paths.append(path)
    return paths


def load_artifact_json(run_dir: Path, path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    target = run_dir / path
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def find_artifact_path(
    plugins: dict[str, Any], plugin_id: str, suffix: str
) -> str | None:
    paths = artifact_paths_for_plugin(plugins, plugin_id)
    for path in paths:
        if path.endswith(suffix):
            return path
    return None


def claim_id(seed: str) -> str:
    return f"claim_{stable_hash(seed):08x}"


def ensure_modeled_fields(plugins: dict[str, Any]) -> list[str]:
    required = {
        "modeled_assumptions",
        "modeled_scope",
        "baseline_host_count",
        "modeled_host_count",
        "baseline_value",
        "modeled_value",
        "delta_value",
        "unit",
    }
    errors: list[str] = []
    for plugin_id, payload in plugins.items():
        findings = payload.get("findings") if isinstance(payload, dict) else None
        if not isinstance(findings, list):
            continue
        for idx, item in enumerate(findings):
            if not isinstance(item, dict):
                continue
            if item.get("measurement_type") != "modeled":
                continue
            missing = [key for key in required if key not in item]
            if missing:
                errors.append(
                    f"{plugin_id}:finding[{idx}] missing {', '.join(sorted(missing))}"
                )
    return errors


def compute_artifact_manifest(
    run_dir: Path, entries: list[dict[str, Any]], schema_version: str = "v2"
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for entry in entries:
        path = entry.get("path")
        if not isinstance(path, str):
            continue
        target = run_dir / path
        if not target.exists():
            continue
        manifest.append(
            {
                "path": path,
                "sha256": file_sha256(target),
                "schema_version": schema_version,
                "source_plugins": entry.get("source_plugins") or [],
                "created_at_utc": entry.get("created_at_utc") or "",
            }
        )
    return manifest


def serialize_rows(
    rows: list[dict[str, Any]], headers: list[str]
) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for row in rows:
        serialized.append({key: row.get(key, "") for key in headers})
    return serialized


def filter_excluded_processes(
    items: list[dict[str, Any]], known_payload: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if not known_payload:
        return items
    exclusions = known_payload.get("recommendation_exclusions")
    if not isinstance(exclusions, dict):
        return items
    processes = exclusions.get("processes")
    if not isinstance(processes, (list, tuple, set)):
        return items
    excluded = {str(p).strip().lower() for p in processes if str(p).strip()}
    if not excluded:
        return items
    filtered: list[dict[str, Any]] = []
    for item in items:
        action_type = str(item.get("action_type") or item.get("action") or "").strip().lower()
        target = item.get("target")
        if isinstance(target, str) and target.strip().lower() in excluded:
            # Fail-closed, but allow certain "system levers" even when the target process
            # is excluded (e.g., QEMAIL/QPEC exclusions for Quorum should not suppress
            # schedule/capacity levers that are explicitly called out as known issues).
            allow = {"add_server", "tune_schedule"}
            if action_type in allow:
                filtered.append(item)
                continue
            # For excluded processes, only keep items that are not directly "about" that
            # process. Most other action types are too noisy to surface when excluded.
            continue
        filtered.append(item)
    return filtered


def redact_if_needed(value: str, enable_redaction: bool) -> str:
    if not enable_redaction:
        return value
    if not value:
        return value
    token = stable_hash(value)
    return f"redacted_{token:08x}"
