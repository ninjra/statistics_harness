from __future__ import annotations


class _StubStorage:
    def fetch_plugin_results(self, _run_id: str):
        return []


class _StubCtx:
    run_id = "stub"
    storage = _StubStorage()
    dataset_version_id = None
    settings: dict = {}


def test_action_search_mip_batched_scheduler_plugin_is_wired() -> None:
    from plugins.analysis_action_search_mip_batched_scheduler_v1.plugin import Plugin

    res = Plugin().run(_StubCtx())
    assert res.status in {"ok", "skipped", "degraded"}
