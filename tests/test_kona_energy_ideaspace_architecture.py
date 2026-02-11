from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from statistic_harness.core.report_v2_utils import filter_excluded_processes
from statistic_harness.core.stat_plugins.registry import run_plugin
from statistic_harness.cli import cmd_payout_report


def test_energy_ebm_scoring_is_deterministic(run_dir: Path) -> None:
    df = pd.DataFrame(
        {
            "PROCESS_ID": ["A", "A", "B", "QEMAIL", "QEMAIL"],
            "QUEUE_DT": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:10Z",
                "2026-01-01T00:00:20Z",
                "2026-01-01T00:00:30Z",
                "2026-01-01T00:05:30Z",
            ],
            "START_DT": [
                "2026-01-01T00:00:05Z",
                "2026-01-01T00:00:12Z",
                "2026-01-01T00:00:25Z",
                "2026-01-01T00:00:40Z",
                "2026-01-01T00:05:40Z",
            ],
            "END_DT": [
                "2026-01-01T00:00:08Z",
                "2026-01-01T00:00:20Z",
                "2026-01-01T00:00:40Z",
                "2026-01-01T00:00:55Z",
                "2026-01-01T00:05:55Z",
            ],
        }
    )

    from tests.conftest import make_context

    ctx = make_context(run_dir, df, settings={"baseline_path": ""}, run_seed=123)
    r1 = run_plugin("analysis_ideaspace_energy_ebm_v1", ctx)
    r2 = run_plugin("analysis_ideaspace_energy_ebm_v1", ctx)

    assert r1.status == "ok"
    assert r2.status == "ok"
    f1 = r1.findings[0]
    f2 = r2.findings[0]
    assert f1.get("energy_total") == f2.get("energy_total")
    assert f1.get("top_terms") == f2.get("top_terms")


def test_verifier_ranks_by_delta_energy_then_confidence(run_dir: Path) -> None:
    # Seed minimal prerequisite artifacts.
    art_action = run_dir / "artifacts" / "analysis_ideaspace_action_planner"
    art_energy = run_dir / "artifacts" / "analysis_ideaspace_energy_ebm_v1"
    art_action.mkdir(parents=True, exist_ok=True)
    art_energy.mkdir(parents=True, exist_ok=True)

    actions = [
        {
            "lever_id": "tune_schedule_qemail_frequency_v1",
            "title": "Tune QEMAIL schedule frequency",
            "action": "Increase QEMAIL interval",
            "confidence": 0.6,
            "evidence": {"metrics": {}},
        },
        {
            "lever_id": "add_qpec_capacity_plus_one_v1",
            "title": "Add QPEC capacity (+1)",
            "action": "Add QPEC+1",
            "confidence": 0.9,
            "evidence": {"metrics": {"qpec_host_count": 2, "qpec_host_entity_keys": ["ALL"]}},
        },
    ]
    (art_action / "recommendations.json").write_text(json.dumps(actions), encoding="utf-8")

    energy = {
        "schema_version": "v1",
        "ideal_mode": "baseline",
        "weights": {"queue_delay_p95": 2.0, "duration_p95": 1.0, "background_overhead_per_min": 1.0},
        "entities": [
            {
                "entity_key": "ALL",
                "entity_label": "ALL",
                "observed": {"queue_delay_p95": 100.0, "duration_p95": 200.0, "background_overhead_per_min": 6.0},
                "ideal": {"queue_delay_p95": 50.0, "duration_p95": 100.0, "background_overhead_per_min": 1.0},
            }
        ],
    }
    (art_energy / "energy_state_vector.json").write_text(json.dumps(energy), encoding="utf-8")

    df = pd.DataFrame({"x": [1, 2, 3]})
    from tests.conftest import make_context

    ctx = make_context(run_dir, df, settings={"max_findings": 10}, run_seed=1, populate=True)
    res = run_plugin("analysis_ebm_action_verifier_v1", ctx)
    assert res.status in {"ok", "skipped"}
    if not res.findings:
        return

    # Ensure sort keys are respected: delta_energy desc then confidence desc then lever_id asc.
    scored = [(float(f.get("delta_energy") or 0.0), float(f.get("confidence") or 0.0)) for f in res.findings]
    assert scored == sorted(scored, key=lambda t: (-t[0], -t[1]))


def test_lever_qemail_frequency_detects_5_minute_schedule(run_dir: Path) -> None:
    fixture = Path("tests/fixtures/qemail_frequency_5min.csv")
    df = pd.read_csv(fixture)
    from tests.conftest import make_context

    ctx = make_context(
        run_dir,
        df,
        settings={
            # Keep the unit test small; evidence gate is configurable.
            "qemail_min_samples": 10,
            "qemail_median_interval_trigger_s": 360.0,
        },
        run_seed=7,
    )
    res = run_plugin("analysis_ideaspace_action_planner", ctx)
    assert res.status == "ok"
    recos_path = run_dir / "artifacts" / "analysis_ideaspace_action_planner" / "recommendations.json"
    assert recos_path.exists()
    recos = json.loads(recos_path.read_text(encoding="utf-8"))
    assert any(r.get("lever_id") == "tune_schedule_qemail_frequency_v1" for r in recos)


def test_filter_excluded_processes_allows_add_server_and_tune_schedule() -> None:
    known_payload = {"recommendation_exclusions": {"processes": ["qemail"]}}
    items = [
        {"action_type": "tune_schedule", "target": "qemail"},
        {"action_type": "add_server", "target": "qemail"},
        {"action_type": "review", "target": "qemail"},
        {"action_type": "review", "target": "other"},
    ]
    out = filter_excluded_processes(items, known_payload)
    assert {"action_type": "tune_schedule", "target": "qemail"} in out
    assert {"action_type": "add_server", "target": "qemail"} in out
    assert {"action_type": "review", "target": "qemail"} not in out
    assert {"action_type": "review", "target": "other"} in out


def test_payout_report_batch_cli_merges_multiple_inputs(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "out"
    a = Path("tests/fixtures/payout_multi_input_a.csv")
    b = Path("tests/fixtures/payout_multi_input_b.csv")

    # Ensure auth isn't required in tests.
    monkeypatch.delenv("STAT_HARNESS_ENABLE_AUTH", raising=False)
    cmd_payout_report([str(a), str(b)], str(out_dir))

    json_path = out_dir / "artifacts" / "report_payout_report_v1" / "payout_report.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    # Per-source breakdown should exist.
    assert payload.get("per_source")
    sources = {row.get("source") for row in payload.get("per_source") or [] if isinstance(row, dict)}
    assert "payout_multi_input_a.csv" in sources
    assert "payout_multi_input_b.csv" in sources
