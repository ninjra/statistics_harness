from __future__ import annotations

import time
import json
from pathlib import Path


def apply_retention(appdata_root: Path, runs_root: Path, *, days: int = 60) -> None:
    """Best-effort retention enforcement.

    - Deletes upload CAS blobs older than `days`.
    - Deletes run directories older than `days` unless a `PINNED` file exists in the run dir.
    - Keeps SQLite metadata (DB is the system-of-record).
    """

    keep_seconds = max(1, int(days)) * 86400
    cutoff = time.time() - keep_seconds

    blobs_dir = appdata_root / "uploads" / "blobs"
    if blobs_dir.exists():
        for blob in blobs_dir.iterdir():
            if not blob.is_file():
                continue
            try:
                if blob.stat().st_mtime < cutoff:
                    blob.unlink()
            except Exception:
                continue

    if runs_root.exists():
        for run_dir in runs_root.iterdir():
            if not run_dir.is_dir():
                continue
            if run_dir.name == "_staging":
                continue
            pinned = run_dir / "PINNED"
            if pinned.exists():
                continue
            try:
                if run_dir.stat().st_mtime >= cutoff:
                    continue
            except Exception:
                continue

            # After retention window: keep the run directory only if it is recommended
            # (or manually pinned via PINNED).
            if _has_recommendations(run_dir / "report.json"):
                continue
            _rmtree_best_effort(run_dir)


def _rmtree_best_effort(path: Path) -> None:
    try:
        for child in path.iterdir():
            try:
                if child.is_dir():
                    _rmtree_best_effort(child)
                else:
                    child.unlink()
            except Exception:
                continue
        try:
            path.rmdir()
        except Exception:
            pass
    except Exception:
        pass


def _has_recommendations(report_path: Path) -> bool:
    if not report_path.exists() or not report_path.is_file():
        return False
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    recs = payload.get("recommendations")
    return isinstance(recs, list) and len(recs) > 0
