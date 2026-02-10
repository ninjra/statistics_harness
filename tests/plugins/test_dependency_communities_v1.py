from __future__ import annotations


class _StubCtx:
    dataset_version_id = None
    settings: dict = {}


def test_dependency_communities_plugins_are_wired() -> None:
    from plugins.analysis_dependency_community_louvain_v1.plugin import Plugin as Louvain
    from plugins.analysis_dependency_community_leiden_v1.plugin import Plugin as Leiden

    res1 = Louvain().run(_StubCtx())
    res2 = Leiden().run(_StubCtx())
    assert res1.status == "skipped"
    assert res2.status == "skipped"
    assert "transform_normalize_mixed" in res1.summary
    assert "transform_normalize_mixed" in res2.summary

