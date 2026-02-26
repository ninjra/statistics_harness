from __future__ import annotations

from scripts.run_loaded_dataset_full import _resolve_exclude_processes


def test_resolve_excludes_merge_mode() -> None:
    values, source = _resolve_exclude_processes(["LOS*"], ["qemail", "los*"], "merge")
    assert values == ["LOS*", "qemail"]
    assert source == "explicit+historical_dataset"


def test_resolve_excludes_explicit_only_mode() -> None:
    values, source = _resolve_exclude_processes(["LOS*"], ["qemail"], "explicit_only")
    assert values == ["LOS*"]
    assert source == "explicit"


def test_resolve_excludes_none_mode() -> None:
    values, source = _resolve_exclude_processes(["LOS*"], ["qemail"], "none")
    assert values == []
    assert source == "none"
