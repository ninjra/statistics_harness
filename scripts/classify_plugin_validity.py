#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_status(value: Any) -> str:
    return str(value or "").strip().lower()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _to_number(value: Any) -> float | None:
    if not _is_number(value):
        return None
    return float(value)


def _max_positive(values: list[Any]) -> float | None:
    best: float | None = None
    for value in values:
        number = _to_number(value)
        if number is None:
            continue
        if best is None or number > best:
            best = number
    return best


def _round_or_none(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _try_git_commit(root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    token = str(proc.stdout or "").strip()
    return token or None


def _reason_code_from_explanations(explanations: dict[str, dict[str, Any]], plugin_id: str) -> str:
    row = explanations.get(plugin_id, {})
    return str(row.get("reason_code") or "").strip()


def _extract_explanation_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    recs = report.get("recommendations")
    if not isinstance(recs, dict):
        return {}
    block = recs.get("explanations")
    if not isinstance(block, dict):
        return {}
    items = block.get("items")
    if not isinstance(items, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in items:
        if not isinstance(row, dict):
            continue
        plugin_id = str(row.get("plugin_id") or "").strip()
        if plugin_id and plugin_id not in out:
            out[plugin_id] = row
    return out


def _extract_recommendation_map(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    recs = report.get("recommendations")
    if not isinstance(recs, dict):
        return {}
    items = recs.get("items")
    if not isinstance(items, list):
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for row in items:
        if not isinstance(row, dict):
            continue
        plugin_id = str(row.get("plugin_id") or "").strip()
        if not plugin_id:
            continue
        out.setdefault(plugin_id, []).append(row)
    return out


def _extract_plugin_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    plugins = report.get("plugins")
    if not isinstance(plugins, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for plugin_id, payload in plugins.items():
        key = str(plugin_id or "").strip()
        if not key or not isinstance(payload, dict):
            continue
        out[key] = payload
    return out


def _first_reason_code_from_findings(findings: list[dict[str, Any]]) -> str:
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        reason = str(finding.get("reason_code") or "").strip()
        if reason:
            return reason
    return ""


def _window_metric_from_rows(rows: list[dict[str, Any]], fallback_rows: list[dict[str, Any]], delta_keys: list[str], gain_keys: list[str]) -> dict[str, float | None]:
    delta = _max_positive([row.get(key) for key in delta_keys for row in rows])
    if delta is None:
        delta = _max_positive([row.get(key) for key in delta_keys for row in fallback_rows])
    gain = _max_positive([row.get(key) for key in gain_keys for row in rows])
    if gain is None:
        gain = _max_positive([row.get(key) for key in gain_keys for row in fallback_rows])
    return {
        "delta_hours": _round_or_none(delta, 2),
        "efficiency_gain_pct": _round_or_none(gain, 3),
    }


def _extract_findings(plugin_payload: dict[str, Any]) -> list[dict[str, Any]]:
    findings = plugin_payload.get("findings")
    if not isinstance(findings, list):
        return []
    return [row for row in findings if isinstance(row, dict)]


def _extract_downstream_consumers(plugin_payload: dict[str, Any], explanation_row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key in ("downstream_consumers", "downstream_plugins"):
        raw = plugin_payload.get(key)
        if isinstance(raw, list):
            for token in raw:
                text = str(token or "").strip()
                if text:
                    out.append(text)
        raw2 = explanation_row.get(key)
        if isinstance(raw2, list):
            for token in raw2:
                text = str(token or "").strip()
                if text:
                    out.append(text)
    deduped: list[str] = []
    seen: set[str] = set()
    for token in out:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


_MARKER_ABSENT_REASON_CODES = {
    "NO_ACTIONABLE_FINDING_CLASS",
    "NO_ACTIONABLE_RESULT",
    "NO_DECISION_SIGNAL",
    "NO_FINDINGS",
    "NO_MODELED_CAPACITY_GAIN",
    "NO_REVENUE_COMPRESSION_PRESSURE",
    "NO_STATISTICAL_SIGNAL",
    "PLUGIN_PRECONDITION_UNMET",
    "PREREQUISITE_UNMET",
    "SHARE_SHIFT_BELOW_THRESHOLD",
}

_TARGETING_FAIL_REASON_CODES = {
    "ADAPTER_RULE_MISSING",
    "NO_DIRECT_PROCESS_TARGET",
    "NO_ROUTING_RULE_MATCH",
}

_FAIL_LOGIC_STATUSES = {"aborted", "degraded", "error", "skipped"}


def _classify_plugin_row(
    *,
    plugin_id: str,
    plugin_row: dict[str, Any],
    plugin_payload: dict[str, Any],
    explanation_row: dict[str, Any],
    recommendation_rows: list[dict[str, Any]],
    run_seed: int,
    dataset_hash: str | None,
    plugin_registry_hash: str,
    plugin_manifest_schema_version: str,
    git_commit: str | None,
) -> dict[str, Any]:
    findings = _extract_findings(plugin_payload)
    report_status = _normalize_status(plugin_payload.get("status"))
    status = report_status or _normalize_status(plugin_row.get("result_status")) or _normalize_status(plugin_row.get("execution_status"))
    reason_code = _reason_code_from_explanations({plugin_id: explanation_row}, plugin_id) or _first_reason_code_from_findings(findings)
    reason_code = str(reason_code or "").strip()
    has_error_payload = bool(plugin_payload.get("error"))

    targeting_status = "unknown"
    raw_targeting = plugin_payload.get("targeting_status")
    if str(raw_targeting or "").strip().lower() in {"pass", "fail", "unknown"}:
        targeting_status = str(raw_targeting).strip().lower()
    if reason_code in _TARGETING_FAIL_REASON_CODES:
        targeting_status = "fail"

    streaming_status = "unknown"
    raw_streaming = plugin_payload.get("streaming_contract_status")
    if str(raw_streaming or "").strip().lower() in {"pass", "fail", "unknown"}:
        streaming_status = str(raw_streaming).strip().lower()
    if plugin_payload.get("streaming_contract_mismatch") is True:
        streaming_status = "fail"
    if plugin_payload.get("streaming_contract_match") is True:
        streaming_status = "pass"

    logic_status = "pass"
    fail_logic_reason = ""
    if status in _FAIL_LOGIC_STATUSES:
        logic_status = "fail"
        fail_logic_reason = f"STATUS_{status.upper()}"
    elif status == "ok" and has_error_payload:
        logic_status = "fail"
        fail_logic_reason = "OK_WITH_ERROR_PAYLOAD"
    elif status == "na" and not reason_code:
        logic_status = "fail"
        fail_logic_reason = "NA_REASON_MISSING"

    marker_status = "unknown"
    if findings:
        marker_status = "present"
    elif status == "na" or reason_code in _MARKER_ABSENT_REASON_CODES:
        marker_status = "absent"

    metrics = {
        "accounting_month": _window_metric_from_rows(
            recommendation_rows,
            findings,
            [
                "modeled_delta_hours_accounting_month",
                "delta_hours_accounting_month",
                "modeled_delta_hours_full_month",
            ],
            [
                "modeled_efficiency_gain_pct_accounting_month",
                "efficiency_gain_pct_accounting_month",
            ],
        ),
        "close_static": _window_metric_from_rows(
            recommendation_rows,
            findings,
            [
                "delta_hours_close_static",
                "modeled_delta_hours_close_static",
            ],
            [
                "efficiency_gain_pct_close_static",
                "modeled_efficiency_gain_pct_close_static",
            ],
        ),
        "close_dynamic": _window_metric_from_rows(
            recommendation_rows,
            findings,
            [
                "delta_hours_close_dynamic",
                "modeled_delta_hours_close_cycle",
                "modeled_delta_hours_close_dynamic",
            ],
            [
                "efficiency_gain_pct_close_dynamic",
                "modeled_efficiency_gain_pct_close_cycle",
                "modeled_efficiency_gain_pct_close_dynamic",
            ],
        ),
        "overall": _window_metric_from_rows(
            recommendation_rows,
            findings,
            ["modeled_delta_hours", "delta_hours"],
            ["modeled_efficiency_gain_pct", "efficiency_gain_pct"],
        ),
    }

    primary_delta = metrics["close_dynamic"]["delta_hours"]
    if primary_delta is None:
        primary_delta = metrics["overall"]["delta_hours"]

    state = "PASS_VALID_MARKER_ABSENT"
    state_reason = reason_code or "NO_ACTIONABLE_SIGNAL"
    if logic_status == "fail":
        state = "FAIL_LOGIC"
        state_reason = fail_logic_reason or reason_code or "LOGIC_VALIDATION_FAILED"
    elif targeting_status == "fail":
        state = "FAIL_TARGETING"
        state_reason = reason_code or "TARGETING_CONTRACT_FAILED"
    elif streaming_status == "fail":
        state = "FAIL_STREAMING_CONTRACT"
        state_reason = reason_code or "STREAMING_CONTRACT_FAILED"
    elif _is_number(primary_delta) and float(primary_delta) >= 1.0:
        state = "PASS_ACTIONABLE"
        state_reason = reason_code or "MATERIAL_IMPACT_GE_1H"
    elif _is_number(primary_delta) and float(primary_delta) > 0.0:
        state = "PASS_VALID_LOW_SIGNAL"
        state_reason = reason_code or "LOW_SIGNAL_LT_1H"
    elif marker_status == "absent":
        state = "PASS_VALID_MARKER_ABSENT"
        state_reason = reason_code or "MARKER_ABSENT"

    downstream_consumers = _extract_downstream_consumers(plugin_payload, explanation_row)
    serializable_for_hash = {
        "plugin_id": plugin_id,
        "state": state,
        "reason_code": state_reason,
        "targeting_status": targeting_status,
        "streaming_contract_status": streaming_status,
        "logic_status": logic_status,
        "marker_status": marker_status,
        "metrics": metrics,
        "downstream_consumers": downstream_consumers,
        "status": status,
    }
    serialized_output_hash = hashlib.sha256(
        json.dumps(serializable_for_hash, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()

    return {
        "plugin_id": plugin_id,
        "state": state,
        "reason_code": state_reason,
        "targeting_status": targeting_status,
        "streaming_contract_status": streaming_status,
        "logic_status": logic_status,
        "marker_status": marker_status,
        "downstream_consumers": downstream_consumers,
        "metrics": metrics,
        "evidence": {
            "run_seed": int(run_seed),
            "dataset_hash": dataset_hash,
            "plugin_registry_hash": plugin_registry_hash,
            "schema_version": plugin_manifest_schema_version,
            "git_commit": git_commit,
            "serialized_output_hash": serialized_output_hash,
        },
    }


def classify_plugin_validity(root: Path, run_id: str, plugin_manifest: dict[str, Any]) -> dict[str, Any]:
    run_dir = root / "appdata" / "runs" / run_id
    report = _load_json(run_dir / "report.json")
    plugin_map = _extract_plugin_map(report)
    explanation_map = _extract_explanation_map(report)
    recommendation_map = _extract_recommendation_map(report)
    plugins = plugin_manifest.get("plugins") if isinstance(plugin_manifest.get("plugins"), list) else []
    git_commit = _try_git_commit(root)
    plugin_rows: list[dict[str, Any]] = []
    for row in plugins:
        if not isinstance(row, dict):
            continue
        plugin_id = str(row.get("plugin_id") or "").strip()
        if not plugin_id:
            continue
        plugin_payload = plugin_map.get(plugin_id, {})
        explanation_row = explanation_map.get(plugin_id, {})
        recommendation_rows = recommendation_map.get(plugin_id, [])
        plugin_rows.append(
            _classify_plugin_row(
                plugin_id=plugin_id,
                plugin_row=row,
                plugin_payload=plugin_payload,
                explanation_row=explanation_row,
                recommendation_rows=recommendation_rows,
                run_seed=int(plugin_manifest.get("run_seed") or 0),
                dataset_hash=(
                    str(plugin_manifest.get("dataset_hash") or "").strip() or None
                ),
                plugin_registry_hash=str(plugin_manifest.get("plugin_registry_hash") or ""),
                plugin_manifest_schema_version=str(plugin_manifest.get("schema_version") or ""),
                git_commit=git_commit,
            )
        )

    plugin_rows = sorted(plugin_rows, key=lambda item: str(item.get("plugin_id") or ""))
    counts = Counter(str(row.get("state") or "") for row in plugin_rows)
    payload = {
        "schema_version": "plugin_validation_contract.v1",
        "generated_at": _now_iso(),
        "run_id": run_id,
        "run_seed": int(plugin_manifest.get("run_seed") or 0),
        "dataset_hash": str(plugin_manifest.get("dataset_hash") or "").strip() or None,
        "plugin_registry_hash": str(plugin_manifest.get("plugin_registry_hash") or ""),
        "git_commit": git_commit,
        "plugin_count": int(len(plugin_rows)),
        "classification_counts": {
            key: int(counts.get(key, 0))
            for key in (
                "PASS_ACTIONABLE",
                "PASS_VALID_LOW_SIGNAL",
                "PASS_VALID_MARKER_ABSENT",
                "FAIL_TARGETING",
                "FAIL_STREAMING_CONTRACT",
                "FAIL_LOGIC",
            )
        },
        "plugins": plugin_rows,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify per-plugin validity state from canonical plugin manifest and run report."
    )
    parser.add_argument("--root", default=str(_repo_root()))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest-json", default="")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    run_id = str(args.run_id)
    manifest_path = (
        Path(args.manifest_json).resolve()
        if str(args.manifest_json).strip()
        else root / "appdata" / "runs" / run_id / "plugin_run_manifest.json"
    )
    if not manifest_path.exists():
        raise SystemExit(f"Missing plugin manifest: {manifest_path}")
    plugin_manifest = _load_json(manifest_path)
    payload = classify_plugin_validity(root, run_id, plugin_manifest)
    out_path = (
        Path(args.out_json).resolve()
        if str(args.out_json).strip()
        else root / "appdata" / "runs" / run_id / "audit" / "plugin_validity_contract.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    fail_count = int(
        payload["classification_counts"]["FAIL_TARGETING"]
        + payload["classification_counts"]["FAIL_STREAMING_CONTRACT"]
        + payload["classification_counts"]["FAIL_LOGIC"]
    )
    print(f"run_id={payload.get('run_id')}")
    print(f"plugin_count={payload.get('plugin_count')}")
    print(f"fail_count={fail_count}")
    print(f"out_json={out_path}")
    if args.strict and fail_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
