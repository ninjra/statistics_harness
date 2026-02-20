from __future__ import annotations

from pathlib import Path

from statistic_harness.core.frozen_surfaces import build_surface_record, save_contract
from statistic_harness.core.pipeline import Pipeline
from statistic_harness.core.plugin_manager import PluginManager


def _plugin_rows(pipeline: Pipeline, run_id: str) -> dict[str, dict]:
    return {row["plugin_id"]: row for row in pipeline.storage.fetch_plugin_results(run_id)}


def _run_and_lock(tmp_path: Path) -> tuple[Pipeline, str, Path]:
    appdata = tmp_path / "appdata"
    pipeline = Pipeline(appdata, Path("plugins"))
    run_id = pipeline.run(
        Path("tests/fixtures/synth_linear.csv"),
        ["profile_basic"],
        {},
        1337,
        force=True,
        reuse_cache=False,
    )
    rows = _plugin_rows(pipeline, run_id)
    profile = rows["profile_basic"]
    manager = PluginManager(Path("plugins"))
    spec = {s.plugin_id: s for s in manager.discover()}["profile_basic"]
    record = build_surface_record(
        spec,
        manager,
        code_hash=str(profile.get("code_hash") or "") or None,
        settings_hash=str(profile.get("settings_hash") or "") or None,
    )
    contract_path = tmp_path / "docs" / "frozen_plugin_surfaces.contract.json"
    save_contract(
        contract_path,
        {
            "source_run_id": run_id,
            "plugins": {"profile_basic": record},
        },
    )
    return pipeline, run_id, contract_path


def test_frozen_surface_enforce_blocks_drift(tmp_path: Path, monkeypatch) -> None:
    pipeline, run_id, contract_path = _run_and_lock(tmp_path)
    row = pipeline.storage.fetch_run(run_id)
    assert row is not None
    dataset_version_id = str(row.get("dataset_version_id") or "")
    assert dataset_version_id

    monkeypatch.setenv("STAT_HARNESS_FROZEN_SURFACES_MODE", "enforce")
    monkeypatch.setenv("STAT_HARNESS_FROZEN_SURFACES_PATH", str(contract_path))

    original = pipeline._spec_code_hash

    def patched(spec):
        value = original(spec)
        if spec.plugin_id == "profile_basic":
            return "forced-frozen-surface-drift"
        return value

    monkeypatch.setattr(pipeline, "_spec_code_hash", patched)
    run2 = pipeline.run(
        None,
        ["profile_basic"],
        {},
        1337,
        force=False,
        reuse_cache=True,
        dataset_version_id=dataset_version_id,
    )
    rows2 = _plugin_rows(pipeline, run2)
    profile2 = rows2["profile_basic"]
    assert str(profile2.get("status") or "").lower() == "error"
    assert "frozen_surface_mismatch" in str(profile2.get("summary") or "")


def test_frozen_surface_warn_allows_execution(tmp_path: Path, monkeypatch) -> None:
    pipeline, run_id, contract_path = _run_and_lock(tmp_path)
    row = pipeline.storage.fetch_run(run_id)
    assert row is not None
    dataset_version_id = str(row.get("dataset_version_id") or "")
    assert dataset_version_id

    monkeypatch.setenv("STAT_HARNESS_FROZEN_SURFACES_MODE", "warn")
    monkeypatch.setenv("STAT_HARNESS_FROZEN_SURFACES_PATH", str(contract_path))

    original = pipeline._spec_code_hash

    def patched(spec):
        value = original(spec)
        if spec.plugin_id == "profile_basic":
            return "forced-frozen-surface-drift"
        return value

    monkeypatch.setattr(pipeline, "_spec_code_hash", patched)
    run2 = pipeline.run(
        None,
        ["profile_basic"],
        {},
        1337,
        force=False,
        reuse_cache=False,
        dataset_version_id=dataset_version_id,
    )
    rows2 = _plugin_rows(pipeline, run2)
    profile2 = rows2["profile_basic"]
    assert str(profile2.get("status") or "").lower() == "ok"
    debug_raw = str(profile2.get("debug_json") or "")
    assert "frozen_surface" in debug_raw

