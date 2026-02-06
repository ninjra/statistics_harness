from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from plugins.analysis_ideaspace_action_planner.plugin import Plugin as ActionPlugin
from plugins.analysis_ideaspace_normative_gap.plugin import Plugin as GapPlugin


class _StorageStub:
    def __init__(self, findings: list[dict]) -> None:
        self._findings = findings

    def fetch_plugin_results(self, run_id: str):
        return [
            {
                "plugin_id": "analysis_ideaspace_normative_gap",
                "findings_json": json.dumps(self._findings),
            }
        ]


class _Ctx:
    def __init__(self, tmp_path: Path, df: pd.DataFrame, settings: dict, storage=None):
        self.run_id = "r1"
        self.run_dir = tmp_path
        self.settings = settings
        self.run_seed = 1337
        self.storage = storage
        self.dataset_version_id = None

        def _load():
            return df

        self.dataset_loader = _load

    def artifacts_dir(self, plugin_id: str) -> Path:
        path = self.run_dir / "artifacts" / plugin_id
        path.mkdir(parents=True, exist_ok=True)
        return path


def test_ideaspace_normative_gap_and_action_planner(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "process_id": ["proc_a"] * 30 + ["proc_b"] * 30 + ["qlos_bad"] * 30,
            "eligible_time": pd.date_range("2026-01-01", periods=90, freq="min"),
            "start_time": list(pd.date_range("2026-01-01 00:10", periods=30, freq="min"))
            + list(pd.date_range("2026-01-01 03:00", periods=30, freq="min"))
            + list(pd.date_range("2026-01-01 01:00", periods=30, freq="min")),
        }
    )
    gap = GapPlugin().run(
        _Ctx(
            tmp_path,
            df,
            {"exclude_processes": ["*los*"], "min_samples": 10},
        )
    )
    assert gap.status == "ok"
    assert all("los" not in str(item.get("process_id")) for item in gap.findings)

    action = ActionPlugin().run(
        _Ctx(
            tmp_path,
            df,
            {"exclude_processes": ["*los*"], "min_runs": 10},
            storage=_StorageStub(gap.findings),
        )
    )
    assert action.status in {"ok", "skipped"}
    for item in action.findings:
        if item.get("kind") == "ideaspace_action" and item.get("process_id"):
            assert "los" not in item["process_id"]
