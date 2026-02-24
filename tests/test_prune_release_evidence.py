from __future__ import annotations

import sqlite3
from pathlib import Path

import scripts.prune_release_evidence as retention_mod


def _init_state_db(db_path: Path) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.executescript(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                dataset_version_id TEXT,
                created_at TEXT,
                status TEXT
            );
            """
        )
        con.executemany(
            "INSERT INTO runs(run_id,dataset_version_id,created_at,status) VALUES (?,?,?,?)",
            [
                ("ds1_run1", "ds1", "2026-02-01T00:00:00+00:00", "completed"),
                ("ds1_run2", "ds1", "2026-02-02T00:00:00+00:00", "completed"),
                ("ds1_run3", "ds1", "2026-02-03T00:00:00+00:00", "completed"),
                ("ds1_run4", "ds1", "2026-02-04T00:00:00+00:00", "completed"),
                ("ds2_run1", "ds2", "2026-02-03T00:00:00+00:00", "completed"),
                ("ds2_run2", "ds2", "2026-02-04T00:00:00+00:00", "partial"),
            ],
        )
        con.commit()
    finally:
        con.close()


def test_plan_prunes_files_referencing_old_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(retention_mod, "ROOT", tmp_path)
    db_path = tmp_path / "state.sqlite"
    evidence_dir = tmp_path / "docs" / "release_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    _init_state_db(db_path)

    keep_name = "hotspots_ds1_run4.md"
    prune_name = "hotspots_ds1_run1.md"
    cross_name = "diff_ds1_run1_to_ds1_run4.json"
    unscoped_name = "golden_release_summary.json"
    for name in (keep_name, prune_name, cross_name, unscoped_name):
        (evidence_dir / name).write_text("x", encoding="utf-8")

    plan = retention_mod.build_plan(
        evidence_dir=evidence_dir,
        db_path=db_path,
        keep_per_dataset=2,
        pin_run_ids=[],
        pin_files=[],
    )
    prune_files = {Path(row["file"]).name for row in plan["prune_files"]}
    keep_files = {Path(row["file"]).name for row in plan["keep_files"]}
    assert keep_name in keep_files
    assert unscoped_name in keep_files
    assert prune_name in prune_files
    assert cross_name in prune_files


def test_pin_run_id_keeps_cross_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(retention_mod, "ROOT", tmp_path)
    db_path = tmp_path / "state.sqlite"
    evidence_dir = tmp_path / "docs" / "release_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    _init_state_db(db_path)

    cross_name = "diff_ds1_run1_to_ds1_run4.json"
    (evidence_dir / cross_name).write_text("x", encoding="utf-8")

    plan = retention_mod.build_plan(
        evidence_dir=evidence_dir,
        db_path=db_path,
        keep_per_dataset=2,
        pin_run_ids=["ds1_run1"],
        pin_files=[],
    )
    prune_files = {Path(row["file"]).name for row in plan["prune_files"]}
    assert cross_name not in prune_files


def test_apply_plan_deletes_pruned_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(retention_mod, "ROOT", tmp_path)
    db_path = tmp_path / "state.sqlite"
    evidence_dir = tmp_path / "docs" / "release_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    _init_state_db(db_path)

    old_path = evidence_dir / "hotspots_ds1_run1.md"
    new_path = evidence_dir / "hotspots_ds1_run4.md"
    old_path.write_text("old", encoding="utf-8")
    new_path.write_text("new", encoding="utf-8")

    plan = retention_mod.build_plan(
        evidence_dir=evidence_dir,
        db_path=db_path,
        keep_per_dataset=1,
        pin_run_ids=[],
        pin_files=[],
    )
    result = retention_mod.apply_plan(plan)
    assert int(result["failed_count"] or 0) == 0
    assert old_path.exists() is False
    assert new_path.exists() is True
