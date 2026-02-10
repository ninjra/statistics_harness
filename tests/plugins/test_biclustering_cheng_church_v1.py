from __future__ import annotations


class _StubCtx:
    dataset_version_id = None
    settings: dict = {}


def test_biclustering_cheng_church_plugin_is_wired() -> None:
    from plugins.analysis_biclustering_cheng_church_v1.plugin import Plugin

    res = Plugin().run(_StubCtx())
    assert res.status == "skipped"
    assert "transform_normalize_mixed" in res.summary

