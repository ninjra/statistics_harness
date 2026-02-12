from __future__ import annotations

import json
from pathlib import Path

from statistic_harness.core.four_pillars import build_four_pillars_scorecard
from statistic_harness.core.pipeline import Pipeline


def _synthetic_report() -> dict:
    return {
        "plugins": {
            "analysis_traceability_manifest_v2": {
                "status": "ok",
                "summary": "ok",
                "findings": [
                    {
                        "measurement_type": "measured",
                        "evidence": {"dataset_id": "d", "dataset_version_id": "dv"},
                    }
                ],
            },
            "report_decision_bundle_v2": {
                "status": "ok",
                "summary": "ok",
                "findings": [
                    {
                        "measurement_type": "modeled",
                        "evidence": {"dataset_id": "d", "dataset_version_id": "dv"},
                    }
                ],
            },
            "analysis_recommendation_dedupe_v2": {
                "status": "ok",
                "summary": "ok",
                "findings": [],
            },
            "analysis_some_other_method": {
                "status": "ok",
                "summary": "ok",
                "findings": [
                    {
                        "measurement_type": "measured",
                        "evidence": {"dataset_id": "d", "dataset_version_id": "dv"},
                    }
                ],
            },
        },
        "recommendations": {
            "known": {
                "items": [
                    {"status": "confirmed"},
                    {"status": "confirmed"},
                    {"status": "below_min"},
                ]
            },
            "discovery": {"items": [{"modeled_percent": 12.5}, {"title": "plain recommendation"}]},
        },
        "lineage": {
            "run": {"run_fingerprint": "abc123"},
            "plugins": {
                "analysis_traceability_manifest_v2": {"execution_fingerprint": "e1"},
                "report_decision_bundle_v2": {"execution_fingerprint": "e2"},
                "analysis_recommendation_dedupe_v2": {"execution_fingerprint": "e3"},
                "analysis_some_other_method": {"execution_fingerprint": "e4"},
            },
        },
        "hotspots": {"top_by_max_rss_kb": [{"max_rss_kb": 256000}]},
    }


def test_four_pillars_scorecard_shape_and_bounds() -> None:
    scorecard = build_four_pillars_scorecard(
        _synthetic_report(),
        run_row={
            "created_at": "2026-02-10T14:00:00+00:00",
            "completed_at": "2026-02-10T14:18:00+00:00",
        },
    )
    assert scorecard["scale"] == "0.0-4.0"
    assert "pillars" in scorecard
    for name in ("performant", "accurate", "secure", "citable"):
        value = float(scorecard["pillars"][name]["score_0_4"])
        assert 0.0 <= value <= 4.0
    overall = float(scorecard["summary"]["overall_0_4"])
    assert 0.0 <= overall <= 4.0


def test_four_pillars_vetoes_tradeoff_imbalance() -> None:
    report = _synthetic_report()
    report["recommendations"] = {"known": {"items": []}, "discovery": {"items": []}}
    scorecard = build_four_pillars_scorecard(report, balance_max_spread=0.5, min_floor=3.0)
    vetoes = scorecard["balance"]["vetoes"]
    codes = {str(v.get("code")) for v in vetoes if isinstance(v, dict)}
    assert "pillar_floor_breach" in codes or "pillar_imbalance" in codes
    assert scorecard["summary"]["vetoed"] is True


def test_four_pillars_not_kona_only() -> None:
    report = _synthetic_report()
    report["plugins"].pop("analysis_some_other_method")
    # No ideaspace plugin in the report on purpose; scorecard must still compute.
    scorecard = build_four_pillars_scorecard(report)
    assert "performant" in scorecard["pillars"]
    assert "accurate" in scorecard["pillars"]
    assert "secure" in scorecard["pillars"]
    assert "citable" in scorecard["pillars"]


def test_pipeline_report_includes_four_pillars(tmp_path, monkeypatch) -> None:
    appdata = tmp_path / "appdata"
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata))
    pipeline = Pipeline(appdata, Path("plugins"))
    run_id = pipeline.run(Path("tests/fixtures/synth_linear.csv"), ["profile_basic"], {}, 42)
    report_path = appdata / "runs" / run_id / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "four_pillars" in report
    assert "pillars" in report["four_pillars"]
    report_md = (appdata / "runs" / run_id / "report.md").read_text(encoding="utf-8")
    assert "### 4-Pillar Scorecard" in report_md
