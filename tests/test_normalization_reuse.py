from __future__ import annotations

from pathlib import Path

from statistic_harness.core.pipeline import Pipeline


def _plugin_rows(pipeline: Pipeline, run_id: str) -> dict[str, dict]:
    return {row["plugin_id"]: row for row in pipeline.storage.fetch_plugin_results(run_id)}


def test_normalization_reused_for_db_only_runs(tmp_path: Path) -> None:
    appdata = tmp_path / "appdata"
    pipeline = Pipeline(appdata, Path("plugins"))

    # Initial run ingests and normalizes.
    run1 = pipeline.run(
        Path("tests/fixtures/synth_linear.csv"),
        ["profile_basic"],
        {},
        42,
        force=True,
        reuse_cache=False,
    )
    run1_row = pipeline.storage.fetch_run(run1)
    assert run1_row is not None
    dataset_version_id = str(run1_row.get("dataset_version_id") or "")
    assert dataset_version_id

    # DB-only rerun should reuse normalization and pure/read-only plugin outputs.
    run2 = pipeline.run(
        None,
        ["profile_basic"],
        {},
        42,
        force=False,
        reuse_cache=True,
        dataset_version_id=dataset_version_id,
    )
    rows2 = _plugin_rows(pipeline, run2)
    assert "transform_normalize_mixed" in rows2
    assert "REUSED" in str(rows2["transform_normalize_mixed"].get("summary") or "")
    assert "profile_basic" in rows2
    assert "REUSED" in str(rows2["profile_basic"].get("summary") or "")
    # Report bundle is run-scoped and must not be reused.
    assert "report_bundle" in rows2
    assert "REUSED" not in str(rows2["report_bundle"].get("summary") or "")


def test_normalization_fingerprint_invalidates_downstream_reuse(
    tmp_path: Path,
    monkeypatch,
) -> None:
    appdata = tmp_path / "appdata"
    pipeline = Pipeline(appdata, Path("plugins"))

    run1 = pipeline.run(
        Path("tests/fixtures/synth_linear.csv"),
        ["profile_basic"],
        {},
        42,
        force=True,
        reuse_cache=False,
    )
    run1_row = pipeline.storage.fetch_run(run1)
    assert run1_row is not None
    dataset_version_id = str(run1_row.get("dataset_version_id") or "")
    assert dataset_version_id

    # Warm cache on DB-only run.
    run2 = pipeline.run(
        None,
        ["profile_basic"],
        {},
        42,
        force=False,
        reuse_cache=True,
        dataset_version_id=dataset_version_id,
    )
    rows2 = _plugin_rows(pipeline, run2)
    assert "REUSED" in str(rows2["transform_normalize_mixed"].get("summary") or "")
    assert "REUSED" in str(rows2["profile_basic"].get("summary") or "")

    # Simulate normalization logic change: force a different normalization code fingerprint.
    original = pipeline._spec_code_hash

    def patched(spec):
        value = original(spec)
        if spec.plugin_id == "transform_normalize_mixed":
            return "forced-normalization-fingerprint-change"
        return value

    monkeypatch.setattr(pipeline, "_spec_code_hash", patched)

    run3 = pipeline.run(
        None,
        ["profile_basic"],
        {},
        42,
        force=False,
        reuse_cache=True,
        dataset_version_id=dataset_version_id,
    )
    rows3 = _plugin_rows(pipeline, run3)

    # Normalization should not be reused under changed fingerprint.
    assert "REUSED" not in str(rows3["transform_normalize_mixed"].get("summary") or "")
    # Downstream cache keys are salted with normalization fingerprint, so profile should re-execute too.
    assert "REUSED" not in str(rows3["profile_basic"].get("summary") or "")
