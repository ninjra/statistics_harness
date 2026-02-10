from __future__ import annotations


class _StubCtx:
    dataset_version_id = None
    settings: dict = {}


def test_similarity_graph_spectral_clustering_plugin_is_wired() -> None:
    from plugins.analysis_similarity_graph_spectral_clustering_v1.plugin import Plugin

    res = Plugin().run(_StubCtx())
    assert res.status == "skipped"
    assert "transform_normalize_mixed" in res.summary

