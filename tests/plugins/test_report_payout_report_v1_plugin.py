from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from plugins.report_payout_report_v1.plugin import Plugin, _payout_report_columns


class _DummyStorage:
    def __init__(self, columns: list[str]) -> None:
        self._columns = list(columns)

    def connection(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def fetch_dataset_columns(self, _dataset_version_id: str, _conn) -> list[dict[str, str]]:
        return [{"original_name": name, "safe_name": f"c{i+1}"} for i, name in enumerate(self._columns)]


def test_payout_report_columns_selects_targeted_subset() -> None:
    storage = _DummyStorage(
        [
            "PROCESS_ID",
            "PARAM_DESCR_LIST",
            "QUEUE_DT",
            "START_DT",
            "END_DT",
            "__source_file",
            "BIG_TEXT_BLOB",
            "UNRELATED_COL",
        ]
    )
    ctx = SimpleNamespace(storage=storage, dataset_version_id="dv1")
    cols = _payout_report_columns(ctx)
    assert "PROCESS_ID" in cols
    assert "PARAM_DESCR_LIST" in cols
    assert "START_DT" in cols
    assert "END_DT" in cols
    assert "__source_file" in cols
    assert "BIG_TEXT_BLOB" not in cols
    assert "UNRELATED_COL" not in cols


def test_payout_report_columns_fallbacks_to_bounded_prefix_when_schema_is_opaque() -> None:
    opaque_cols = [f"c{i}" for i in range(60)]
    storage = _DummyStorage(opaque_cols)
    ctx = SimpleNamespace(storage=storage, dataset_version_id="dv1")
    cols = _payout_report_columns(ctx)
    assert cols == opaque_cols[:32]


def test_plugin_uses_selected_columns_loader_path(tmp_path) -> None:
    all_columns = [
        "PROCESS_ID",
        "PARAM_DESCR_LIST",
        "QUEUE_DT",
        "START_DT",
        "END_DT",
        "__source_file",
        "BIG_TEXT_BLOB",
    ]
    storage = _DummyStorage(all_columns)
    captured: dict[str, list[str] | None] = {"columns": None}

    def dataset_loader(columns=None, row_limit=None):
        captured["columns"] = list(columns) if isinstance(columns, list) else None
        return pd.DataFrame(
            {
                "PROCESS_ID": ["JBPREPAY", "OTHER"],
                "PARAM_DESCR_LIST": ["P1", "P2"],
                "QUEUE_DT": ["2026-01-01 10:00:00.000", "2026-01-01 10:01:00.000"],
                "START_DT": ["2026-01-01 10:00:05.000", "2026-01-01 10:01:05.000"],
                "END_DT": ["2026-01-01 10:00:20.000", "2026-01-01 10:01:20.000"],
                "__source_file": ["a.csv", "a.csv"],
            }
        )

    ctx = SimpleNamespace(
        storage=storage,
        dataset_version_id="dv1",
        settings={},
        dataset_loader=dataset_loader,
        artifacts_dir=lambda _plugin_id: tmp_path,
        run_dir=tmp_path,
    )
    result = Plugin().run(ctx)
    assert result.status == "ok"
    requested = captured["columns"] or []
    assert requested
    assert "BIG_TEXT_BLOB" not in requested
    assert "PROCESS_ID" in requested


def test_plugin_returns_ok_when_dataset_has_no_columns(tmp_path) -> None:
    storage = _DummyStorage([])
    called = {"loader": 0}

    def dataset_loader(columns=None, row_limit=None):
        called["loader"] += 1
        return pd.DataFrame()

    ctx = SimpleNamespace(
        storage=storage,
        dataset_version_id="dv1",
        settings={},
        dataset_loader=dataset_loader,
        artifacts_dir=lambda _plugin_id: tmp_path,
        run_dir=tmp_path,
    )
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert called["loader"] == 0
