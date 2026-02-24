from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from scripts.run_loaded_dataset_full import (
    _plugins_with_capability,
    _route_preflight,
    _sql_assist_preflight,
)


class _DummyStorage:
    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - deterministic shim
        self.args = args
        self.kwargs = kwargs


def test_plugins_with_capability_detects_sql_assist_required() -> None:
    plugins = [
        "transform_sql_intents_pack_v1",
        "transform_sqlpack_materialize_v1",
        "analysis_param_near_duplicate_minhash_v1",
    ]
    required = _plugins_with_capability(plugins, "sql_assist_required")
    assert "transform_sql_intents_pack_v1" in required
    assert "transform_sqlpack_materialize_v1" in required
    assert "analysis_param_near_duplicate_minhash_v1" not in required


def test_route_preflight_blocks_when_required_plugin_missing(tmp_path: Path) -> None:
    report = _route_preflight(
        plugin_ids=["analysis_ideaspace_action_planner"],
        route_enable=True,
        output_dir=tmp_path,
    )
    assert report["route_enable"] is True
    assert int(report["blocking_count"] or 0) == 1
    assert report["missing_plugins"] == ["analysis_ebm_action_verifier_v1"]


def test_route_preflight_no_block_when_route_disabled(tmp_path: Path) -> None:
    report = _route_preflight(
        plugin_ids=[],
        route_enable=False,
        output_dir=tmp_path,
    )
    assert report["route_enable"] is False
    assert int(report["blocking_count"] or 0) == 0
    assert report["required_plugins"] == []


def test_sql_assist_preflight_passes_with_schema_snapshot(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "scripts.run_loaded_dataset_full._plugins_with_capability",
        lambda _plugin_ids, _capability: ["transform_sql_intents_pack_v1"],
    )
    fake_schema = SimpleNamespace(schema_hash="hash123", snapshot={"tables": [{"name": "t"}]})
    monkeypatch.setitem(
        sys.modules,
        "statistic_harness.core.storage",
        SimpleNamespace(Storage=_DummyStorage),
    )
    monkeypatch.setitem(
        sys.modules,
        "statistic_harness.core.sql_schema_snapshot",
        SimpleNamespace(snapshot_schema=lambda _storage: fake_schema),
    )
    report = _sql_assist_preflight(
        db_path=tmp_path / "state.sqlite",
        plugin_ids=["transform_sql_intents_pack_v1"],
        output_dir=tmp_path,
    )
    assert int(report["checked_count"] or 0) == 1
    assert report["schema_ready"] is True
    assert int(report["blocking_count"] or 0) == 0


def test_sql_assist_preflight_blocks_when_schema_unavailable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "scripts.run_loaded_dataset_full._plugins_with_capability",
        lambda _plugin_ids, _capability: ["transform_sql_intents_pack_v1"],
    )

    def _raise(_storage):  # pragma: no cover - simple deterministic branch
        raise RuntimeError("schema unavailable")

    monkeypatch.setitem(
        sys.modules,
        "statistic_harness.core.storage",
        SimpleNamespace(Storage=_DummyStorage),
    )
    monkeypatch.setitem(
        sys.modules,
        "statistic_harness.core.sql_schema_snapshot",
        SimpleNamespace(snapshot_schema=_raise),
    )
    report = _sql_assist_preflight(
        db_path=tmp_path / "state.sqlite",
        plugin_ids=["transform_sql_intents_pack_v1"],
        output_dir=tmp_path,
    )
    assert int(report["checked_count"] or 0) == 1
    assert report["schema_ready"] is False
    assert int(report["blocking_count"] or 0) == 1
