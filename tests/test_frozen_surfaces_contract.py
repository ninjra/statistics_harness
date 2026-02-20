from __future__ import annotations

from pathlib import Path

from statistic_harness.core.frozen_surfaces import (
    build_surface_record,
    contract_plugin_map,
    evaluate_locked_surface,
    load_contract,
    save_contract,
)
from statistic_harness.core.plugin_manager import PluginManager


def _spec_and_manager() -> tuple[object, PluginManager]:
    manager = PluginManager(Path("plugins"))
    spec_map = {spec.plugin_id: spec for spec in manager.discover()}
    return spec_map["profile_basic"], manager


def test_build_surface_record_is_stable_for_same_spec() -> None:
    spec, manager = _spec_and_manager()
    one = build_surface_record(spec, manager)
    two = build_surface_record(spec, manager)
    assert one["surface_hash"] == two["surface_hash"]
    assert one["plugin_id"] == "profile_basic"
    assert one["plugin_version"]
    assert one["code_hash"]
    assert one["settings_hash"]


def test_contract_roundtrip(tmp_path: Path) -> None:
    spec, manager = _spec_and_manager()
    rec = build_surface_record(spec, manager)
    path = tmp_path / "docs" / "frozen_plugin_surfaces.contract.json"
    save_contract(
        path,
        {
            "source_run_id": "run_x",
            "plugins": {"profile_basic": rec},
        },
    )
    loaded = load_contract(path)
    plugins = contract_plugin_map(loaded)
    assert loaded["schema"] == "frozen_surfaces_contract.v1"
    assert "profile_basic" in plugins
    assert plugins["profile_basic"]["surface_hash"] == rec["surface_hash"]


def test_evaluate_locked_surface() -> None:
    ok = evaluate_locked_surface(
        plugin_id="profile_basic",
        expected_surface_hash="abc",
        actual_surface_hash="abc",
    )
    bad = evaluate_locked_surface(
        plugin_id="profile_basic",
        expected_surface_hash="abc",
        actual_surface_hash="def",
    )
    assert ok["ok"] is True
    assert bad["ok"] is False

