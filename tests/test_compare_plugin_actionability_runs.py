from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.compare_plugin_actionability_runs import _latest_run_for_dataset, compare_runs


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_runs_emits_novelty_sets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    before = {
        "lineage": {"dataset": {"dataset_version_id": "dataset_a"}},
        "plugins": {"p1": {}, "p2": {}},
        "recommendations": {
            "items": [
                {
                    "plugin_id": "p1",
                    "kind": "k1",
                    "action_type": "a1",
                    "recommendation": "Do A",
                    "scope_class": "general",
                    "where": {"process": "x"},
                }
            ],
            "known": {"items": []},
            "discovery": {
                "items": [
                    {
                        "plugin_id": "p1",
                        "kind": "k1",
                        "action_type": "a1",
                        "recommendation": "Do A",
                        "scope_class": "general",
                        "where": {"process": "x"},
                    }
                ]
            },
            "explanations": {"items": [{"plugin_id": "p2"}]},
        },
    }
    after = {
        "lineage": {"dataset": {"dataset_version_id": "dataset_b"}},
        "plugins": {"p1": {}, "p2": {}},
        "recommendations": {
            "items": [
                {
                    "plugin_id": "p1",
                    "kind": "k1",
                    "action_type": "a1",
                    "recommendation": "Do A",
                    "scope_class": "general",
                    "where": {"process": "x"},
                },
                {
                    "plugin_id": "p1",
                    "kind": "k2",
                    "action_type": "a2",
                    "recommendation": "Do B",
                    "scope_class": "general",
                    "where": {"process": "y"},
                },
            ],
            "known": {"items": []},
            "discovery": {
                "items": [
                    {
                        "plugin_id": "p1",
                        "kind": "k1",
                        "action_type": "a1",
                        "recommendation": "Do A",
                        "scope_class": "general",
                        "where": {"process": "x"},
                    },
                    {
                        "plugin_id": "p1",
                        "kind": "k2",
                        "action_type": "a2",
                        "recommendation": "Do B",
                        "scope_class": "general",
                        "where": {"process": "y"},
                    },
                ]
            },
            "explanations": {"items": [{"plugin_id": "p2"}]},
        },
    }
    _write_report(tmp_path / "appdata" / "runs" / "before_run" / "report.json", before)
    _write_report(tmp_path / "appdata" / "runs" / "after_run" / "report.json", after)

    result = compare_runs("before_run", "after_run")

    assert result["before_dataset_version_id"] == "dataset_a"
    assert result["after_dataset_version_id"] == "dataset_b"
    assert result["novelty"]["discovery"]["new_count"] == 1
    assert result["novelty"]["discovery"]["unchanged_count"] == 1
    assert result["novelty"]["discovery"]["dropped_count"] == 0


def test_latest_run_for_dataset_uses_completed_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    state = tmp_path / "appdata" / "state.sqlite"
    state.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(state))
    try:
        conn.execute(
            "CREATE TABLE runs (run_id TEXT, dataset_version_id TEXT, status TEXT, created_at TEXT)"
        )
        conn.execute(
            "INSERT INTO runs (run_id, dataset_version_id, status, created_at) VALUES (?, ?, ?, ?)",
            ("older", "dataset_x", "completed", "2026-02-16T00:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO runs (run_id, dataset_version_id, status, created_at) VALUES (?, ?, ?, ?)",
            ("newer_partial", "dataset_x", "partial", "2026-02-17T00:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO runs (run_id, dataset_version_id, status, created_at) VALUES (?, ?, ?, ?)",
            ("newer_completed", "dataset_x", "completed", "2026-02-17T01:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()

    assert _latest_run_for_dataset("dataset_x") == "newer_completed"
