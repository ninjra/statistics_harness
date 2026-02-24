#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "appdata" / "state.sqlite"
DEFAULT_EVIDENCE_DIR = ROOT / "docs" / "release_evidence"


@dataclass(frozen=True)
class RunRow:
    run_id: str
    dataset_version_id: str
    created_at: str
    status: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def _load_runs(db_path: Path) -> list[RunRow]:
    if not db_path.exists():
        return []
    con = _connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT run_id, dataset_version_id, created_at, status
            FROM runs
            WHERE run_id IS NOT NULL
            ORDER BY created_at DESC
            """
        ).fetchall()
    finally:
        con.close()
    out: list[RunRow] = []
    for row in rows:
        run_id = str(row["run_id"] or "").strip()
        if not run_id:
            continue
        out.append(
            RunRow(
                run_id=run_id,
                dataset_version_id=str(row["dataset_version_id"] or "").strip(),
                created_at=str(row["created_at"] or "").strip(),
                status=str(row["status"] or "").strip().lower(),
            )
        )
    return out


def _keep_runs_by_dataset(rows: list[RunRow], keep_per_dataset: int) -> set[str]:
    grouped: dict[str, list[RunRow]] = defaultdict(list)
    for row in rows:
        status = row.status
        if status not in {"completed", "partial", "running"}:
            continue
        grouped[row.dataset_version_id].append(row)
    keep_ids: set[str] = set()
    limit = max(1, int(keep_per_dataset))
    for dataset_version_id, members in grouped.items():
        _ = dataset_version_id  # appease linters; key is useful for readability.
        for row in members[:limit]:
            keep_ids.add(row.run_id)
    return keep_ids


def _candidate_files(evidence_dir: Path) -> list[Path]:
    if not evidence_dir.exists():
        return []
    return sorted([p for p in evidence_dir.rglob("*") if p.is_file()], key=lambda p: str(p).lower())


def _referenced_run_ids(filename: str, all_run_ids: list[str]) -> list[str]:
    found = [run_id for run_id in all_run_ids if run_id in filename]
    return sorted(set(found))


def build_plan(
    *,
    evidence_dir: Path,
    db_path: Path,
    keep_per_dataset: int,
    pin_run_ids: list[str],
    pin_files: list[str],
) -> dict[str, Any]:
    rows = _load_runs(db_path)
    all_run_ids = sorted({row.run_id for row in rows}, key=len, reverse=True)
    keep_run_ids = _keep_runs_by_dataset(rows, keep_per_dataset)
    keep_run_ids.update(str(v).strip() for v in pin_run_ids if str(v).strip())
    pin_file_names = {str(v).strip() for v in pin_files if str(v).strip()}

    files = _candidate_files(evidence_dir)
    keep_files: list[dict[str, Any]] = []
    prune_files: list[dict[str, Any]] = []

    for path in files:
        name = path.name
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        refs = _referenced_run_ids(name, all_run_ids)
        decision: str
        reason: str
        if name in pin_file_names:
            decision = "keep"
            reason = "pinned_file"
        elif not refs:
            decision = "keep"
            reason = "unscoped_file"
        else:
            missing = [rid for rid in refs if rid not in keep_run_ids]
            if missing:
                decision = "prune"
                reason = "references_pruned_run_ids"
            else:
                decision = "keep"
                reason = "references_kept_run_ids"
        item = {
            "file": rel,
            "name": name,
            "referenced_run_ids": refs,
            "reason": reason,
        }
        if decision == "prune":
            prune_files.append(item)
        else:
            keep_files.append(item)

    by_dataset: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.run_id in keep_run_ids:
            by_dataset[row.dataset_version_id].append(row.run_id)
    for dataset_version_id, run_ids in by_dataset.items():
        by_dataset[dataset_version_id] = sorted(set(run_ids))

    return {
        "schema_version": "release_evidence_retention_plan.v1",
        "generated_at_utc": _now_iso(),
        "db_path": str(db_path),
        "evidence_dir": str(evidence_dir),
        "keep_per_dataset": int(keep_per_dataset),
        "pin_run_ids": sorted(str(v).strip() for v in pin_run_ids if str(v).strip()),
        "pin_files": sorted(pin_file_names),
        "stats": {
            "run_count": int(len(rows)),
            "known_run_id_count": int(len(all_run_ids)),
            "kept_run_id_count": int(len(keep_run_ids)),
            "file_count": int(len(files)),
            "keep_file_count": int(len(keep_files)),
            "prune_file_count": int(len(prune_files)),
        },
        "kept_runs_by_dataset": dict(sorted(by_dataset.items(), key=lambda kv: kv[0])),
        "keep_files": keep_files,
        "prune_files": prune_files,
    }


def apply_plan(plan: dict[str, Any]) -> dict[str, Any]:
    deleted: list[str] = []
    missing: list[str] = []
    failed: list[dict[str, str]] = []
    for item in plan.get("prune_files") or []:
        if not isinstance(item, dict):
            continue
        rel = str(item.get("file") or "").strip()
        if not rel:
            continue
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
            continue
        try:
            path.unlink()
            deleted.append(rel)
        except Exception as exc:  # pragma: no cover
            failed.append({"file": rel, "error": str(exc)})
    return {
        "deleted_files": deleted,
        "missing_files": missing,
        "failed_files": failed,
        "deleted_count": int(len(deleted)),
        "missing_count": int(len(missing)),
        "failed_count": int(len(failed)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prune old docs/release_evidence files with a rolling per-dataset retention policy."
    )
    parser.add_argument("--state-db", default=str(DEFAULT_DB))
    parser.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR))
    parser.add_argument("--keep-per-dataset", type=int, default=3)
    parser.add_argument("--pin-run-id", action="append", default=[])
    parser.add_argument("--pin-file", action="append", default=["openplanter_pack_release_gate.json"])
    parser.add_argument("--out-json", default="")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    db_path = Path(str(args.state_db))
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    evidence_dir = Path(str(args.evidence_dir))
    if not evidence_dir.is_absolute():
        evidence_dir = ROOT / evidence_dir

    plan = build_plan(
        evidence_dir=evidence_dir,
        db_path=db_path,
        keep_per_dataset=int(args.keep_per_dataset),
        pin_run_ids=[str(v) for v in (args.pin_run_id or [])],
        pin_files=[str(v) for v in (args.pin_file or [])],
    )
    if bool(args.apply):
        plan["apply"] = apply_plan(plan)
    out_json = str(args.out_json).strip()
    if out_json:
        out_path = Path(out_json)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 1 if bool(plan.get("apply", {}).get("failed_count")) else 0


if __name__ == "__main__":
    raise SystemExit(main())
