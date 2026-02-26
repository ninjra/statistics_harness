from __future__ import annotations

from statistic_harness.core.report import _build_recommendations


def _cc_row(process: str, score: float, count: int) -> dict[str, object]:
    return {
        "kind": "close_cycle_contention",
        "process_norm": process,
        "process": process,
        "slowdown_ratio": 1.5,
        "correlation": 0.7,
        "estimated_improvement_pct": score,
        "modeled_reduction_pct": score,
        "modeled_reduction_hours": score * 10.0,
        "close_count": count,
        "open_count": 100,
        "median_duration_close": 15.0,
        "median_duration_open": 10.0,
        "measurement_type": "modeled",
    }


def test_close_cycle_contention_selection_keeps_next_non_excluded_candidates(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_REQUIRE_DIRECT_PROCESS_ACTION", "0")
    monkeypatch.setenv("STAT_HARNESS_EXCLUDED_PROCESSES", "*los*")
    report = {
        "plugins": {
            "analysis_close_cycle_contention": {
                "findings": [
                    _cc_row("losloadcld", 0.60, 2000),
                    _cc_row("losa", 0.55, 1800),
                    _cc_row("losb", 0.52, 1600),
                    _cc_row("proc_a", 0.40, 1500),
                    _cc_row("proc_b", 0.35, 1200),
                    _cc_row("proc_c", 0.32, 1000),
                ]
            }
        }
    }

    payload = _build_recommendations(report)
    discovery = ((payload.get("discovery") or {}).get("items") or [])
    cc_rows = [
        row
        for row in discovery
        if isinstance(row, dict)
        and str(row.get("plugin_id") or "") == "analysis_close_cycle_contention"
        and str(row.get("kind") or "") == "close_cycle_contention"
    ]
    processes = {
        str(
            row.get("primary_process_id")
            or row.get("process_id")
            or (row.get("where") or {}).get("process_norm")
            or ""
        ).strip().lower()
        for row in cc_rows
    }

    assert "proc_a" in processes
    assert "proc_b" in processes
    assert "proc_c" in processes
    assert "losloadcld" not in processes


def test_close_cycle_contention_prefers_modeled_reduction_hours_for_delta(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_REQUIRE_DIRECT_PROCESS_ACTION", "0")
    report = {
        "plugins": {
            "analysis_close_cycle_contention": {
                "findings": [
                    {
                        **_cc_row("proc_modeled", 0.25, 900),
                        "modeled_reduction_hours": 9.0,
                        "median_duration_close": 11.0,
                        "median_duration_open": 10.0,
                    }
                ]
            }
        }
    }

    payload = _build_recommendations(report)
    discovery = ((payload.get("discovery") or {}).get("items") or [])
    row = next(
        (
            item
            for item in discovery
            if isinstance(item, dict)
            and str((item.get("where") or {}).get("process_norm") or "").strip().lower()
            == "proc_modeled"
            and str(item.get("plugin_id") or "") == "analysis_close_cycle_contention"
        ),
        None,
    )
    assert isinstance(row, dict)
    assert row.get("modeled_delta_hours") == 9.0
