from __future__ import annotations

from pathlib import Path

import pandas as pd


class _Ctx:
    def __init__(self, df: pd.DataFrame, run_dir: Path, settings: dict | None = None) -> None:
        self._df = df
        self.run_dir = run_dir
        self.settings = settings or {}
        self.run_seed = 7

    def dataset_loader(self):
        return self._df

    def artifacts_dir(self, plugin_id: str) -> Path:
        path = self.run_dir / "artifacts" / plugin_id
        path.mkdir(parents=True, exist_ok=True)
        return path


def test_changepoint_detection_v1_runs(tmp_path: Path) -> None:
    from plugins.changepoint_detection_v1.plugin import Plugin

    df = pd.DataFrame(
        {
            "y": [1, 1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 10, 11, 10, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10],
        }
    )
    res = Plugin().run(_Ctx(df, tmp_path))
    assert res.status in {"ok", "skipped"}
    if res.status == "ok":
        assert any(f.get("kind") == "changepoint" for f in res.findings)


def test_process_mining_plugins_run(tmp_path: Path) -> None:
    from plugins.process_mining_conformance_v1.plugin import Plugin as ConformancePlugin
    from plugins.process_mining_discovery_v1.plugin import Plugin as DiscoveryPlugin

    df = pd.DataFrame(
        {
            "case_id": ["A", "A", "A", "B", "B", "C", "C"],
            "activity": ["start", "validate", "end", "start", "end", "start", "end"],
            "timestamp": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:01:00Z",
                "2026-01-01T00:02:00Z",
                "2026-01-01T00:00:30Z",
                "2026-01-01T00:01:30Z",
                "2026-01-01T00:00:45Z",
                "2026-01-01T00:01:45Z",
            ],
        }
    )
    dres = DiscoveryPlugin().run(_Ctx(df, tmp_path))
    cres = ConformancePlugin().run(_Ctx(df, tmp_path))
    assert dres.status in {"ok", "skipped"}
    assert cres.status in {"ok", "skipped"}


def test_causal_recommendations_v1_runs(tmp_path: Path) -> None:
    from plugins.causal_recommendations_v1.plugin import Plugin

    df = pd.DataFrame(
        {
            "treatment_signal": [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0],
            "outcome_metric": [1.0, 0.9, 1.1, 2.1, 2.0, 2.2, 2.1, 0.95, 2.05, 1.05, 2.0, 1.0],
        }
    )
    res = Plugin().run(_Ctx(df, tmp_path))
    assert res.status in {"ok", "skipped", "degraded"}
    if res.status == "ok":
        assert any(f.get("kind") == "recommendation" for f in res.findings)
