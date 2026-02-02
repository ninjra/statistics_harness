from statistic_harness.core.storage import Storage
from statistic_harness.core.tenancy import scope_identifier
from statistic_harness.core.utils import now_iso


def test_tenant_isolation_storage(tmp_path):
    db_path = tmp_path / "state.sqlite"
    tenant_a = "tenant_a"
    tenant_b = "tenant_b"
    storage_a = Storage(db_path, tenant_a)
    storage_b = Storage(db_path, tenant_b)

    created_at = now_iso()
    project_a = scope_identifier(tenant_a, "project")
    dataset_a = scope_identifier(tenant_a, "dataset")
    version_a = scope_identifier(tenant_a, "dv")
    storage_a.ensure_project(project_a, project_a, created_at)
    storage_a.ensure_dataset(dataset_a, project_a, dataset_a, created_at)
    storage_a.ensure_dataset_version(
        version_a, dataset_a, created_at, "table_a", "hash_a"
    )

    project_b = scope_identifier(tenant_b, "project")
    dataset_b = scope_identifier(tenant_b, "dataset")
    version_b = scope_identifier(tenant_b, "dv")
    storage_b.ensure_project(project_b, project_b, created_at)
    storage_b.ensure_dataset(dataset_b, project_b, dataset_b, created_at)
    storage_b.ensure_dataset_version(
        version_b, dataset_b, created_at, "table_b", "hash_b"
    )

    assert {row["project_id"] for row in storage_a.list_projects()} == {project_a}
    assert {row["project_id"] for row in storage_b.list_projects()} == {project_b}

    assert {row["dataset_version_id"] for row in storage_a.list_dataset_versions()} == {
        version_a
    }
    assert {row["dataset_version_id"] for row in storage_b.list_dataset_versions()} == {
        version_b
    }


def test_tenant_isolation_uploads_runs(tmp_path):
    db_path = tmp_path / "state.sqlite"
    tenant_a = "tenant_a"
    tenant_b = "tenant_b"
    storage_a = Storage(db_path, tenant_a)
    storage_b = Storage(db_path, tenant_b)

    created_at = now_iso()
    project_a = scope_identifier(tenant_a, "project")
    dataset_a = scope_identifier(tenant_a, "dataset")
    version_a = scope_identifier(tenant_a, "dv")
    storage_a.ensure_project(project_a, project_a, created_at)
    storage_a.ensure_dataset(dataset_a, project_a, dataset_a, created_at)
    storage_a.ensure_dataset_version(
        version_a, dataset_a, created_at, "table_a", "hash_a"
    )

    project_b = scope_identifier(tenant_b, "project")
    dataset_b = scope_identifier(tenant_b, "dataset")
    version_b = scope_identifier(tenant_b, "dv")
    storage_b.ensure_project(project_b, project_b, created_at)
    storage_b.ensure_dataset(dataset_b, project_b, dataset_b, created_at)
    storage_b.ensure_dataset_version(
        version_b, dataset_b, created_at, "table_b", "hash_b"
    )

    upload_id_a = "upload_a"
    upload_id_b = "upload_b"
    storage_a.create_upload(upload_id_a, "a.csv", 10, "sha_a", created_at)
    storage_b.create_upload(upload_id_b, "b.csv", 12, "sha_b", created_at)

    run_id_a = "run_a"
    run_id_b = "run_b"
    storage_a.create_run(
        run_id=run_id_a,
        created_at=created_at,
        status="completed",
        upload_id=upload_id_a,
        input_filename="a.csv",
        canonical_path="path_a",
        settings={},
        error=None,
        run_seed=0,
        project_id=project_a,
        dataset_id=dataset_a,
        dataset_version_id=version_a,
        input_hash="sha_a",
    )
    storage_b.create_run(
        run_id=run_id_b,
        created_at=created_at,
        status="completed",
        upload_id=upload_id_b,
        input_filename="b.csv",
        canonical_path="path_b",
        settings={},
        error=None,
        run_seed=0,
        project_id=project_b,
        dataset_id=dataset_b,
        dataset_version_id=version_b,
        input_hash="sha_b",
    )

    assert storage_a.fetch_upload(upload_id_b) is None
    assert storage_b.fetch_upload(upload_id_a) is None
    assert {row["upload_id"] for row in storage_a.list_uploads()} == {upload_id_a}
    assert {row["upload_id"] for row in storage_b.list_uploads()} == {upload_id_b}

    assert storage_a.fetch_run(run_id_b) is None
    assert storage_b.fetch_run(run_id_a) is None
    assert {row["run_id"] for row in storage_a.list_runs_by_project(project_a)} == {
        run_id_a
    }
    assert {row["run_id"] for row in storage_b.list_runs_by_project(project_b)} == {
        run_id_b
    }
