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


def _base_dataset() -> pd.DataFrame:
    elig_a = pd.date_range("2026-01-01", periods=30, freq="h").tolist() + pd.date_range(
        "2026-02-01", periods=30, freq="h"
    ).tolist()
    elig_b = pd.date_range("2026-01-01", periods=30, freq="h").tolist() + pd.date_range(
        "2026-02-01", periods=30, freq="h"
    ).tolist()
    elig_los = pd.date_range("2026-01-01", periods=30, freq="h").tolist() + pd.date_range(
        "2026-02-01", periods=30, freq="h"
    ).tolist()
    eligible = elig_a + elig_b + elig_los

    starts_a = [ts + pd.Timedelta(minutes=8) for ts in elig_a]
    starts_b = [ts + pd.Timedelta(minutes=80) for ts in elig_b]
    starts_los = [ts + pd.Timedelta(minutes=50) for ts in elig_los]
    starts = starts_a + starts_b + starts_los

    return pd.DataFrame(
        {
            "process_id": ["proc_a"] * 60 + ["proc_b"] * 60 + ["qlos_bad"] * 60,
            "eligible_time": eligible,
            "start_time": starts,
        }
    )


def test_ideaspace_normative_gap_and_action_planner(tmp_path: Path) -> None:
    df = _base_dataset()

    gap = GapPlugin().run(
        _Ctx(
            tmp_path,
            df,
            {"exclude_processes": ["*los*"], "min_samples": 10},
        )
    )
    assert gap.status == "ok"
    assert all("los" not in str(item.get("process_id")) for item in gap.findings)
    assert all(int(item.get("windows") or 0) >= 2 for item in gap.findings)

    action = ActionPlugin().run(
        _Ctx(
            tmp_path,
            df,
            {"exclude_processes": ["*los*"], "min_runs": 10, "min_gap_pct": 0.05},
            storage=_StorageStub(gap.findings),
        )
    )
    assert action.status == "ok"
    for item in action.findings:
        if item.get("kind") == "ideaspace_action" and item.get("process_id"):
            assert "los" not in item["process_id"]
            assert item.get("references")


def test_normative_gap_skips_when_missing_columns(tmp_path: Path) -> None:
    df = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
    result = GapPlugin().run(_Ctx(tmp_path, df, {}))
    assert result.status == "skipped"


def test_normative_gap_deterministic_sampling(tmp_path: Path) -> None:
    df = _base_dataset()
    many = pd.concat([df] * 1000, ignore_index=True)
    ctx = _Ctx(tmp_path, many, {"max_rows": 5000, "min_samples": 10, "exclude_processes": ["*los*"]})
    r1 = GapPlugin().run(ctx)
    r2 = GapPlugin().run(ctx)
    assert r1.status == "ok"
    assert r1.findings == r2.findings
    assert r1.metrics == r2.metrics


def test_action_planner_skip_when_no_source_findings(tmp_path: Path) -> None:
    df = _base_dataset()

    class _EmptyStore:
        def fetch_plugin_results(self, run_id: str):
            return []

    result = ActionPlugin().run(_Ctx(tmp_path, df, {}, storage=_EmptyStore()))
    assert result.status == "skipped"


def test_action_planner_skip_when_gates_fail(tmp_path: Path) -> None:
    findings = [
        {
            "kind": "ideaspace_gap",
            "process_id": "proc_x",
            "runs": 5,
            "windows": 1,
            "gap_sec": 50,
            "gap_pct": 0.01,
            "median_wait_sec": 100,
            "ideal_wait_sec": 90,
        }
    ]
    dummy_df = _base_dataset().head(5)
    result = ActionPlugin().run(_Ctx(tmp_path, dummy_df, {}, storage=_StorageStub(findings)))
    assert result.status == "skipped"
    assert result.findings[0]["measurement_type"] == "not_applicable"
