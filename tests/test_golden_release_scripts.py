from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import scripts.build_golden_release_evidence_pack as evidence_pack
import scripts.compare_dataset_runs as compare_dataset_runs
from scripts.golden_release_delta_map import build_delta_map
from scripts.rank_streaming_offenders import build_ranked


def test_build_delta_map_shape_and_counts() -> None:
    payload = build_delta_map()
    items = list(payload.get("items") or [])
    summary = dict(payload.get("summary") or {})

    assert payload.get("schema_version") == "golden_release_delta_map.v1"
    assert payload.get("source_plan") == "docs/codex_4pillars_golden_release_plan.md"
    assert len(items) >= 5
    assert all(str(item.get("status")) in {"implemented", "partial", "missing"} for item in items)
    assert int(summary.get("implemented", 0)) + int(summary.get("partial", 0)) + int(
        summary.get("missing", 0)
    ) == len(items)


def test_build_ranked_filters_unbounded_and_sorts_by_score() -> None:
    matrix_rows: list[dict[str, Any]] = [
        {"plugin_id": "c", "plugin_type": "analysis", "uses_dataset_loader_unbounded": False},
        {"plugin_id": "a", "plugin_type": "analysis", "uses_dataset_loader_unbounded": True},
        {"plugin_id": "b", "plugin_type": "analysis", "uses_dataset_loader_unbounded": True},
    ]
    exec_stats: dict[str, dict[str, float]] = {
        "a": {"avg_duration_ms": 4_000.0, "max_rss_kb": 1_024.0, "execution_count": 3.0},
        "b": {"avg_duration_ms": 500.0, "max_rss_kb": 8_192.0, "execution_count": 5.0},
    }

    ranked = build_ranked(matrix_rows, exec_stats)
    assert [row["plugin_id"] for row in ranked] == ["b", "a"]
    assert all(bool(row.get("uses_dataset_loader_unbounded")) for row in ranked)
    assert ranked[0]["priority_score"] > ranked[1]["priority_score"]


def test_compare_dataset_runs_main_writes_artifacts(monkeypatch, tmp_path: Path) -> None:
    def _fake_run(cmd: list[str], check: bool) -> None:  # pragma: no cover - invoked by main
        assert check is True
        out_json = Path(cmd[cmd.index("--output-json") + 1])
        payload = {
            "dataset_before": {"dataset_version_id": "before_ds"},
            "dataset_after": {"dataset_version_id": "after_ds"},
            "plugin_counts": {"payload_changed": 2},
            "added_plugins": [],
            "removed_plugins": [],
            "status_changes": [{"plugin_id": "p_status"}],
            "changed_plugins": [
                {
                    "plugin_id": "p_status",
                    "status_before": "ok",
                    "status_after": "failed",
                    "changed_components": ["status"],
                },
                {
                    "plugin_id": "p_payload",
                    "status_before": "ok",
                    "status_after": "ok",
                    "changed_components": ["stats", "recommendations"],
                },
            ],
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    monkeypatch.setattr(compare_dataset_runs.subprocess, "run", _fake_run)
    prev_argv = sys.argv[:]
    try:
        out_dir = tmp_path / "release_evidence"
        sys.argv = [
            "compare_dataset_runs.py",
            "--dataset-before",
            "before_ds",
            "--dataset-after",
            "after_ds",
            "--out-dir",
            str(out_dir),
        ]
        assert compare_dataset_runs.main() == 0
    finally:
        sys.argv = prev_argv

    summary_path = out_dir / "comparison_summary.json"
    deltas_path = out_dir / "comparison_recommendation_deltas.json"
    report_path = out_dir / "comparison_report.md"
    assert summary_path.exists()
    assert deltas_path.exists()
    assert report_path.exists()

    deltas = json.loads(deltas_path.read_text(encoding="utf-8")).get("items") or []
    assert [row.get("plugin_id") for row in deltas] == ["p_status", "p_payload"]
    report = report_path.read_text(encoding="utf-8")
    assert "Dataset Run Comparison" in report
    assert "`before_ds`" in report and "`after_ds`" in report


def test_evidence_pack_helpers_render_expected_counts() -> None:
    counts = evidence_pack._status_counts(  # noqa: SLF001 - helper is intentionally internal
        [
            {"status": "ok"},
            {"status": "ok"},
            {"status": "failed"},
            {"status": ""},
        ]
    )
    assert counts == {"ok": 2, "failed": 1, "unknown": 1}

    md = evidence_pack._render_markdown(  # noqa: SLF001 - helper is intentionally internal
        [
            {
                "run_id": "run_123",
                "status": "completed",
                "dataset_version_id": "dataset_abc",
                "plugin_count": 255,
                "status_counts": {"ok": 250, "skipped": 5},
            }
        ]
    )
    assert "Golden Release Evidence Summary" in md
    assert "`run_123`" in md
    assert "ok:250, skipped:5" in md
