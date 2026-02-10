from __future__ import annotations


class _StubStorage:
    def fetch_plugin_results(self, _run_id: str):
        # No prior actionable ops levers available.
        return []


class _StubCtx:
    # This plugin reads prior plugin results; provide a minimal storage stub.
    run_id = "stub"
    storage = _StubStorage()
    dataset_version_id = None
    settings: dict = {}


def test_action_search_simulated_annealing_plugin_is_wired() -> None:
    from plugins.analysis_action_search_simulated_annealing_v1.plugin import Plugin

    res = Plugin().run(_StubCtx())
    assert res.status in {"ok", "skipped", "degraded"}
