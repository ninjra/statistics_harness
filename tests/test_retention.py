from __future__ import annotations

import os
import time
from pathlib import Path

from statistic_harness.core.retention import apply_retention


def _set_mtime(path: Path, ts: float) -> None:
    os.utime(path, (ts, ts))


def test_apply_retention_prunes_old_blobs_and_runs(tmp_path: Path) -> None:
    appdata = tmp_path / "appdata"
    runs_root = tmp_path / "runs"
    blobs_dir = appdata / "uploads" / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    old_blob = blobs_dir / "old"
    new_blob = blobs_dir / "new"
    old_blob.write_text("x", encoding="utf-8")
    new_blob.write_text("y", encoding="utf-8")

    old_run = runs_root / "run_old"
    recommended_run = runs_root / "run_recommended"
    pinned_run = runs_root / "run_pinned"
    old_run.mkdir()
    recommended_run.mkdir()
    pinned_run.mkdir()
    (pinned_run / "PINNED").write_text("1", encoding="utf-8")
    (recommended_run / "report.json").write_text(
        '{"recommendations":[{"id":"r1"}]}', encoding="utf-8"
    )

    now = time.time()
    two_days_ago = now - (2 * 86400)
    _set_mtime(old_blob, two_days_ago)
    _set_mtime(old_run, two_days_ago)
    _set_mtime(recommended_run, two_days_ago)

    apply_retention(appdata, runs_root, days=1)

    assert not old_blob.exists()
    assert new_blob.exists()
    assert not old_run.exists()
    assert recommended_run.exists()
    assert pinned_run.exists()
