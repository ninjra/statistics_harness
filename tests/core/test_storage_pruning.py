"""Storage pruning tests.

Verifies run retention, cascade deletion, orphan cleanup, vacuum,
and archive functionality.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from statistic_harness.core.storage import Storage
from statistic_harness.core.types import PluginResult
from statistic_harness.core.utils import now_iso


def _create_storage(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "state.sqlite")


def _create_run(storage: Storage, run_id: str, created_at: str) -> None:
    storage.create_run(
        run_id=run_id,
        created_at=created_at,
        status="completed",
        upload_id="test-upload",
        input_filename="test.csv",
        canonical_path="/test.csv",
        settings={},
        error=None,
        run_seed=42,
    )


def _save_result(storage: Storage, run_id: str, plugin_id: str) -> None:
    result = PluginResult(
        status="ok",
        summary="Test result",
        metrics={},
        findings=[],
        artifacts=[],
        error=None,
    )
    storage.save_plugin_result(
        run_id=run_id,
        plugin_id=plugin_id,
        plugin_version="0.1.0",
        executed_at=now_iso(),
        code_hash="abc123",
        settings_hash="def456",
        dataset_hash="ghi789",
        result=result,
    )


def _old_iso(days_ago: int) -> str:
    dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_ago)
    return dt.isoformat()


class TestPruneRunsOlderThanDays:
    def test_prune_removes_old_runs(self, tmp_path: Path) -> None:
        storage = _create_storage(tmp_path)
        _create_run(storage, "old-run", _old_iso(120))
        _save_result(storage, "old-run", "plugin_a")

        deleted = storage.prune_runs_older_than_days(90)
        assert deleted == 1

        with storage.connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            assert count == 0
            results = conn.execute("SELECT COUNT(*) FROM plugin_results_v2").fetchone()[0]
            assert results == 0

    def test_prune_preserves_recent(self, tmp_path: Path) -> None:
        storage = _create_storage(tmp_path)
        _create_run(storage, "old-run", _old_iso(120))
        _create_run(storage, "new-run", now_iso())
        _save_result(storage, "old-run", "plugin_a")
        _save_result(storage, "new-run", "plugin_a")

        deleted = storage.prune_runs_older_than_days(90)
        assert deleted == 1

        with storage.connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            assert count == 1
            row = conn.execute("SELECT run_id FROM runs").fetchone()
            assert row["run_id"] == "new-run"

    def test_prune_no_old_runs_returns_zero(self, tmp_path: Path) -> None:
        storage = _create_storage(tmp_path)
        _create_run(storage, "new-run", now_iso())
        deleted = storage.prune_runs_older_than_days(90)
        assert deleted == 0


class TestArchive:
    def test_archive_creates_json(self, tmp_path: Path) -> None:
        storage = _create_storage(tmp_path)
        _create_run(storage, "old-run", _old_iso(120))

        archive_dir = tmp_path / "archive"
        deleted = storage.prune_runs_older_than_days(
            90, archive=True, archive_dir=archive_dir
        )
        assert deleted == 1
        assert (archive_dir / "old-run.json").exists()

        data = json.loads((archive_dir / "old-run.json").read_text())
        assert data["run_id"] == "old-run"


class TestVacuum:
    def test_vacuum_runs(self, tmp_path: Path) -> None:
        storage = _create_storage(tmp_path)
        _create_run(storage, "run-1", now_iso())
        storage.vacuum()
        assert storage.db_path.exists()


class TestRetentionSummary:
    def test_retention_summary_returns_expected_keys(self, tmp_path: Path) -> None:
        storage = _create_storage(tmp_path)
        _create_run(storage, "run-1", now_iso())
        summary = storage.retention_summary()
        assert summary["total_runs"] == 1
        assert "oldest_run_date" in summary
        assert "db_size_mb" in summary
        assert summary["db_size_mb"] > 0


class TestPruneOrphanedArtifacts:
    def test_prune_orphaned_artifacts(self, tmp_path: Path) -> None:
        storage = _create_storage(tmp_path)
        # Create two runs, add artifacts to both, then delete one run directly
        _create_run(storage, "run-1", now_iso())
        _create_run(storage, "run-2", now_iso())
        storage.upsert_artifact(
            "run-1", "keep.csv", "abc123", 100, "text/csv", now_iso(), "plugin_a"
        )
        storage.upsert_artifact(
            "run-2", "orphan.csv", "def456", 50, "text/csv", now_iso(), "plugin_a"
        )
        # Delete run-2 directly (bypassing cascade) to create orphan
        with storage.connection() as conn:
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("DELETE FROM runs WHERE run_id = ?", ("run-2",))
            conn.execute("PRAGMA foreign_keys = ON")

        removed = storage.prune_orphaned_artifacts()
        assert removed == 1
