from __future__ import annotations

from statistic_harness.core.storage import Storage
from statistic_harness.core.types import PluginResult
from scripts.compare_run_outputs import compare_datasets, compare_runs


def _seed_dataset(storage: Storage, dataset_version_id: str, created_at: str) -> None:
    project_id = f"project_{dataset_version_id}"
    dataset_id = f"dataset_{dataset_version_id}"
    storage.ensure_project(project_id, project_id, created_at)
    storage.ensure_dataset(dataset_id, project_id, dataset_id, created_at)
    storage.ensure_dataset_version(
        dataset_version_id=dataset_version_id,
        dataset_id=dataset_id,
        created_at=created_at,
        table_name=f"dataset_{dataset_version_id}",
        data_hash=f"hash_{dataset_version_id}",
    )


def _seed_run(
    storage: Storage,
    run_id: str,
    dataset_version_id: str,
    created_at: str,
    status: str,
) -> None:
    storage.create_run(
        run_id=run_id,
        created_at=created_at,
        status=status,
        upload_id="local",
        input_filename=f"db://{dataset_version_id}",
        canonical_path=f"/tmp/{run_id}.csv",
        settings={},
        error=None,
        run_seed=123,
        requested_run_seed=123,
        project_id=f"project_{dataset_version_id}",
        dataset_id=f"dataset_{dataset_version_id}",
        dataset_version_id=dataset_version_id,
        input_hash=f"hash_{dataset_version_id}",
    )


def _save_result(
    storage: Storage,
    run_id: str,
    plugin_id: str,
    status: str,
    summary: str,
    findings: list[dict] | None = None,
    debug: dict | None = None,
) -> None:
    storage.save_plugin_result(
        run_id=run_id,
        plugin_id=plugin_id,
        plugin_version="1.0.0",
        executed_at="2026-01-01T00:00:00+00:00",
        code_hash="code_hash",
        settings_hash="settings_hash",
        dataset_hash="dataset_hash",
        result=PluginResult(
            status=status,
            summary=summary,
            metrics={},
            findings=findings or [],
            artifacts=[],
            debug=debug or {},
        ),
    )


def test_compare_runs_reports_status_and_payload_changes(tmp_path) -> None:
    storage = Storage(tmp_path / "state.sqlite")
    dataset = "ds_same"
    _seed_dataset(storage, dataset, "2026-01-01T00:00:00+00:00")
    _seed_run(storage, "run_old", dataset, "2026-01-01T00:00:00+00:00", "completed")
    _seed_run(storage, "run_new", dataset, "2026-01-02T00:00:00+00:00", "completed")

    _save_result(
        storage,
        "run_old",
        "plugin_alpha",
        "degraded",
        "old alpha",
        findings=[{"title": "old finding"}],
    )
    _save_result(
        storage,
        "run_new",
        "plugin_alpha",
        "ok",
        "new alpha",
        findings=[{"title": "new finding"}],
    )
    _save_result(
        storage,
        "run_old",
        "plugin_beta",
        "ok",
        "same summary",
        debug={"source": "old"},
    )
    _save_result(
        storage,
        "run_new",
        "plugin_beta",
        "ok",
        "same summary",
        debug={"source": "new"},
    )
    _save_result(storage, "run_old", "plugin_only_old", "ok", "old only")
    _save_result(storage, "run_new", "plugin_only_new", "ok", "new only")

    out = compare_runs(storage, "run_old", "run_new")

    assert out["plugin_counts"]["before"] == 3
    assert out["plugin_counts"]["after"] == 3
    assert out["plugin_counts"]["added"] == 1
    assert out["plugin_counts"]["removed"] == 1
    assert out["plugin_counts"]["status_changed"] == 1
    assert out["plugin_counts"]["material_payload_changed"] == 1
    assert out["status_changes"] == [
        {"plugin_id": "plugin_alpha", "status_before": "degraded", "status_after": "ok"}
    ]
    assert out["added_plugins"] == ["plugin_only_new"]
    assert out["removed_plugins"] == ["plugin_only_old"]
    assert "plugin_alpha" in out["material_changed_plugins"]
    assert out["component_change_counts"]["findings_json"] == 1
    assert out["component_change_counts"]["debug_json"] == 1


def test_compare_datasets_selects_latest_completed_run_per_dataset(tmp_path) -> None:
    storage = Storage(tmp_path / "state.sqlite")
    ds_before = "ds_before"
    ds_after = "ds_after"
    _seed_dataset(storage, ds_before, "2026-01-01T00:00:00+00:00")
    _seed_dataset(storage, ds_after, "2026-01-01T00:00:00+00:00")

    _seed_run(storage, "run_before_completed", ds_before, "2026-01-01T00:00:00+00:00", "completed")
    _seed_run(storage, "run_before_aborted", ds_before, "2026-01-03T00:00:00+00:00", "aborted")
    _seed_run(storage, "run_after_completed", ds_after, "2026-01-02T00:00:00+00:00", "completed")

    _save_result(storage, "run_before_completed", "plugin_x", "ok", "before")
    _save_result(storage, "run_before_aborted", "plugin_x", "ok", "aborted newer")
    _save_result(storage, "run_after_completed", "plugin_x", "ok", "after")

    out = compare_datasets(storage, ds_before, ds_after, statuses=["completed"])

    assert out["mode"] == "dataset_to_dataset"
    assert out["dataset_before"]["dataset_version_id"] == ds_before
    assert out["dataset_after"]["dataset_version_id"] == ds_after
    assert out["dataset_run_selection"]["run_before"]["run_id"] == "run_before_completed"
    assert out["dataset_run_selection"]["run_after"]["run_id"] == "run_after_completed"
    assert out["run_before"]["run_id"] == "run_before_completed"
    assert out["run_after"]["run_id"] == "run_after_completed"
