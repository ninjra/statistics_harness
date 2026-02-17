from __future__ import annotations

from pathlib import Path

from statistic_harness.core.report import write_report


def test_write_report_trims_plugin_dumps_for_markdown(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("STAT_HARNESS_REPORT_MD_MAX_FINDINGS_PER_PLUGIN", "2")
    monkeypatch.setenv("STAT_HARNESS_REPORT_MD_MAX_STRING_LEN", "40")
    monkeypatch.setenv("STAT_HARNESS_REPORT_MD_MAX_EVIDENCE_IDS", "3")
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": "run_trim",
        "created_at": "2026-02-17T00:00:00+00:00",
        "status": "completed",
        "input": {"rows": 10, "cols": 3},
        "plugins": {
            "analysis_demo": {
                "status": "ok",
                "summary": "x" * 120,
                "metrics": {},
                "findings": [
                    {
                        "kind": "demo",
                        "description": "y" * 120,
                        "evidence": {"row_ids": list(range(10)), "column_ids": ["a", "b", "c", "d"]},
                    },
                    {"kind": "demo2", "description": "z" * 80},
                    {"kind": "demo3", "description": "w" * 80},
                ],
                "artifacts": [],
            }
        },
        "recommendations": {"items": [], "summary": ""},
    }
    write_report(report, run_dir)
    text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "report_dump_truncated" in text
    assert "... (7 more)" in text
    assert "... [truncated" in text
