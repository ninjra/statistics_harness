#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _matrix_map(matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in matrix.get("plugins") or []:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("plugin_id") or "").strip()
        if pid:
            out[pid] = row
    return out


def _manifest_plugin_ids(run_dir: Path) -> list[str]:
    payload = _load_json(run_dir / "run_manifest.json")
    rows = payload.get("plugins")
    if not isinstance(rows, list):
        return []
    out: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("plugin_id") or "").strip()
        if pid:
            out.append(pid)
    return sorted(set(out))


def _runtime_access(run_dir: Path, plugin_id: str) -> dict[str, Any]:
    path = run_dir / "artifacts" / plugin_id / "runtime_access.json"
    payload = _load_json(path)
    data = payload.get("data_access")
    return data if isinstance(data, dict) else {}


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def audit_streaming_contract(*, run_dir: Path, matrix_path: Path) -> dict[str, Any]:
    matrix = _load_json(matrix_path)
    expected = _matrix_map(matrix)
    plugin_ids = _manifest_plugin_ids(run_dir)
    rows: list[dict[str, Any]] = []
    mismatch_count = 0

    for plugin_id in plugin_ids:
        exp = expected.get(plugin_id, {})
        access = _runtime_access(run_dir, plugin_id)
        iter_calls = _as_int(access.get("iter_batches_calls"))
        loader_calls = _as_int(access.get("dataset_loader_calls"))
        loader_unbounded = _as_int(access.get("dataset_loader_unbounded_calls"))
        loader_bounded = _as_int(access.get("dataset_loader_bounded_calls"))

        expected_iter = bool(exp.get("uses_dataset_iter_batches"))
        expected_loader = bool(exp.get("uses_dataset_loader"))
        expected_loader_mode = str(exp.get("dataset_loader_mode") or "none").strip().lower()

        violations: list[str] = []
        if expected_iter and iter_calls <= 0:
            violations.append("EXPECTED_ITER_BATCHES_NOT_USED")
        if not expected_loader and loader_calls > 0:
            violations.append("UNDECLARED_DATASET_LOADER_USED")
        if expected_loader_mode == "bounded" and loader_unbounded > 0:
            violations.append("UNBOUNDED_LOADER_CALL_DETECTED")
        if expected_loader_mode == "none" and loader_calls > 0:
            violations.append("LOADER_CALLS_PRESENT_WHEN_NONE_EXPECTED")
        if expected_loader_mode == "unbounded" and loader_calls > 0 and loader_bounded > 0 and loader_unbounded == 0:
            violations.append("EXPECTED_UNBOUNDED_BUT_ONLY_BOUNDED_CALLS")

        status = "contract_match" if not violations else "contract_mismatch"
        if not access:
            status = "runtime_access_missing"
            violations = ["RUNTIME_ACCESS_ARTIFACT_MISSING"]
        if status != "contract_match":
            mismatch_count += 1
        rows.append(
            {
                "plugin_id": plugin_id,
                "status": status,
                "expected": {
                    "uses_dataset_iter_batches": expected_iter,
                    "uses_dataset_loader": expected_loader,
                    "dataset_loader_mode": expected_loader_mode,
                },
                "actual": {
                    "iter_batches_calls": iter_calls,
                    "dataset_loader_calls": loader_calls,
                    "dataset_loader_bounded_calls": loader_bounded,
                    "dataset_loader_unbounded_calls": loader_unbounded,
                },
                "violations": violations,
            }
        )

    return {
        "schema_version": "streaming_contract_audit.v1",
        "run_dir": str(run_dir),
        "plugin_count": len(rows),
        "mismatch_count": mismatch_count,
        "ok": mismatch_count == 0,
        "plugins": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit runtime streaming/data-access contract by plugin.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", default=str(ROOT / "appdata" / "runs"))
    parser.add_argument("--matrix", default=str(ROOT / "docs" / "plugin_data_access_matrix.json"))
    parser.add_argument("--out-json", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    run_dir = Path(str(args.runs_root)) / str(args.run_id).strip()
    payload = audit_streaming_contract(
        run_dir=run_dir,
        matrix_path=Path(str(args.matrix)),
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out_json = str(args.out_json).strip()
    if out_json:
        out_path = Path(out_json)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if bool(args.strict) and not bool(payload.get("ok")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

