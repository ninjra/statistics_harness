from __future__ import annotations

from pathlib import Path

import pandas as pd

from statistic_harness.core.baselines import schema_sha256, signed_digest
from statistic_harness.core.ideaspace_feature_extractor import pick_columns
from statistic_harness.core.lever_library import build_default_lever_recommendations
from statistic_harness.core.stat_plugins.columns import infer_columns
from statistic_harness.core.utils import now_iso, read_json, write_json

from plugins.analysis_ideaspace_normative_gap.plugin import Plugin as GapPlugin
from plugins.analysis_ideaspace_action_planner.plugin import Plugin as ActionPlugin

from conftest import make_context


def _dt_series_span(n: int, span_s: int, start: str = "2026-01-01T00:00:00Z") -> pd.Series:
    """Return ISO8601 strings spanning exactly span_s seconds (SQLite-safe)."""
    base = pd.Timestamp(start)
    if n <= 1:
        return pd.Series([base.isoformat().replace("+00:00", "Z")])
    offsets = [round(i * span_s / float(n - 1)) for i in range(n)]
    values = [(base + pd.Timedelta(seconds=int(off))).isoformat().replace("+00:00", "Z") for off in offsets]
    return pd.Series(values)


def test_frontier_is_deterministic_with_ties(run_dir: Path):
    # Two frontier points (A better duration, B better throughput), C ties distance to both.
    df = pd.DataFrame(
        {
            "process": (["A"] * 100) + (["B"] * 300) + (["C"] * 50),
            "ts": pd.concat(
                [_dt_series_span(100, 60), _dt_series_span(300, 60), _dt_series_span(50, 60)],
                ignore_index=True,
            ),
            # C is dominated by A, but equidistant (normalized) to A and B.
            "duration": ([10.0] * 100) + ([30.0] * 300) + ([17.8078] * 50),
        }
    )
    ctx = make_context(run_dir, df, settings={})
    result = GapPlugin().run(ctx)
    assert result.status == "ok"
    entities = read_json(run_dir / "artifacts" / "analysis_ideaspace_normative_gap" / "entities_table.json")
    by_key = {e["entity_key"]: e for e in entities}
    assert by_key["process=C"]["ideal_entity_key"] == "process=A"


def test_gap_zero_on_identical_to_reference(run_dir: Path):
    df = pd.DataFrame(
        {
            "process": ["A"] * 250,
            "ts": _dt_series_span(250, 60),
            "duration": [10.0] * 250,
        }
    )
    # Build a signed baseline that matches the observed ALL vector.
    inferred = infer_columns(df, {"time_column": "auto", "group_by": "auto", "value_columns": "auto"})
    cols = pick_columns(df, inferred, {"ideaspace_max_group_cols": 0})
    ideal_vector = {"duration_p95": 10.0, "rate_per_min": 250.0}
    schema_path = Path(__file__).resolve().parents[1] / "docs" / "ideaspace_baseline.schema.json"
    baseline = {
        "baseline_id": "test_baseline",
        "version": "1.0.0",
        "created_at": now_iso(),
        "schema_hash": schema_sha256(schema_path),
        "ideal_vector": ideal_vector,
        "signature": {"algo": "sha256", "digest": ""},
    }
    baseline["signature"]["digest"] = signed_digest(baseline)
    baseline_path = run_dir / "baseline.json"
    write_json(baseline_path, baseline)

    ctx = make_context(run_dir, df, settings={"analysis_ideaspace_normative_gap": {"ideaspace_baseline_path": str(baseline_path)}})
    result = GapPlugin().run(ctx)
    assert result.status == "ok"
    entities = read_json(run_dir / "artifacts" / "analysis_ideaspace_normative_gap" / "entities_table.json")
    all_row = [e for e in entities if e["entity_key"] == "ALL"][0]
    gaps = all_row.get("gaps") or {}
    assert abs(float(gaps.get("duration_p95", 0.0))) <= 1e-9
    assert abs(float(gaps.get("rate_per_min", 0.0))) <= 1e-9


def test_tiered_degradation_no_numeric_still_ok(run_dir: Path):
    df = pd.DataFrame({"process": ["A"] * 50, "note": ["x"] * 50})
    ctx = make_context(run_dir, df, settings={})
    result = GapPlugin().run(ctx)
    assert result.status in {"ok", "skipped"}


def test_normative_gap_degrades_on_degenerate_frontier_with_variance(run_dir: Path):
    n = 220
    df = pd.DataFrame(
        {
            "process": (["A"] * n) + (["B"] * n),
            "ts": pd.concat(
                [
                    _dt_series_span(n, 220, start="2026-01-01T00:00:00Z"),
                    _dt_series_span(n, 60, start="2026-01-01T00:00:00Z"),
                ],
                ignore_index=True,
            ),
            "duration": ([10.0] * n) + ([30.0] * n),
        }
    )
    ctx = make_context(run_dir, df, settings={})
    result = GapPlugin().run(ctx)
    assert result.status == "ok"
    assert any(str(f.get("kind") or "") == "ideaspace_degeneracy" for f in result.findings)

    diag = read_json(run_dir / "artifacts" / "analysis_ideaspace_normative_gap" / "normative_gap_diagnostics.json")
    assert bool(diag.get("degenerate_output")) is True
    freshness = read_json(run_dir / "artifacts" / "analysis_ideaspace_normative_gap" / "freshness.json")
    assert freshness.get("source_run_id") == "test-run"
    assert isinstance(freshness.get("dataset_input_hash"), str)


def test_action_planner_no_recos_when_triggers_not_met(run_dir: Path):
    df = pd.DataFrame({"process": ["A"] * 50, "ts": _dt_series_span(50, 60), "duration": [10.0] * 50})
    ctx = make_context(run_dir, df, settings={})
    result = ActionPlugin().run(ctx)
    assert result.status == "skipped"


def test_reco_priority_isolation_triggered(run_dir: Path):
    n = 250
    eligible = pd.to_datetime(_dt_series_span(n, 60), utc=True)
    start = eligible + pd.to_timedelta([600] * n, unit="s")
    df = pd.DataFrame(
        {
            "eligible_ts": eligible.astype(str),
            "start_ts": start.astype(str),
            "process": ["A"] * n,
        }
    )
    inferred = infer_columns(df, {"time_column": "auto", "group_by": "auto", "value_columns": "auto"})
    cols = pick_columns(df, inferred, {})
    recos = build_default_lever_recommendations(df, cols, {"ideaspace_min_rows_for_reco": 200, "ideaspace_queue_delay_p95_trigger_s": 120.0})
    assert any(r.lever_id == "priority_isolation" for r in recos)


def test_reco_blackout_triggered(run_dir: Path):
    n = 300
    start = pd.Series(["2026-01-01T00:00:00Z"] * n)
    end = pd.Series(["2026-01-01T00:02:00Z"] * n)
    # Mark most as scheduled in peak buckets.
    cron_job = ["cron_cleanup"] * n
    df = pd.DataFrame({"start_ts": start, "end_ts": end, "cron_job": cron_job})
    inferred = infer_columns(df, {"time_column": "auto", "group_by": "auto", "value_columns": "auto"})
    cols = pick_columns(df, inferred, {})
    recos = build_default_lever_recommendations(df, cols, {"ideaspace_blackout_trigger_ratio": 0.25, "ideaspace_bucket_seconds": 60, "ideaspace_min_rows_for_reco": 200})
    assert any(r.lever_id == "blackout_scheduled_jobs" for r in recos)


def test_reco_concurrency_cap_triggered(run_dir: Path):
    # High concurrency early (many intervals overlapping), ends later -> negative corr with ended per bucket.
    n = 400
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    start_dt = pd.to_datetime([base + pd.Timedelta(seconds=i) for i in range(n)], utc=True)
    end_dt = start_dt + pd.to_timedelta([600] * n, unit="s")
    df = pd.DataFrame({"start_ts": start_dt.astype(str), "end_ts": end_dt.astype(str)})
    inferred = infer_columns(df, {"time_column": "auto", "group_by": "auto", "value_columns": "auto"})
    cols = pick_columns(df, inferred, {})
    recos = build_default_lever_recommendations(df, cols, {"ideaspace_concurrency_thrash_corr_trigger": -0.05, "ideaspace_bucket_seconds": 60, "ideaspace_min_rows_for_reco": 200})
    assert any(r.lever_id == "cap_concurrency" for r in recos)


def test_reco_batch_split_triggered(run_dir: Path):
    df = pd.DataFrame(
        {
            "batch_size": ([1] * 220) + ([100] * 60),
            "duration": ([10.0] * 220) + ([1000.0] * 60),
        }
    )
    inferred = infer_columns(df, {"time_column": "auto", "group_by": "auto", "value_columns": "auto"})
    cols = pick_columns(df, inferred, {})
    recos = build_default_lever_recommendations(df, cols, {"ideaspace_batch_tail_multiplier_trigger": 1.5, "ideaspace_min_rows_for_reco": 200})
    assert any(r.lever_id == "split_batches" for r in recos)


def test_reco_retry_backoff_triggered(run_dir: Path):
    n = 250
    ts = pd.Series(["2026-01-01T00:00:00Z"] * n)
    msg = ["timeout error"] * 20 + ["ok"] * (n - 20)
    df = pd.DataFrame({"ts": ts, "message": msg})
    inferred = infer_columns(df, {"time_column": "auto", "group_by": "auto", "value_columns": "auto"})
    cols = pick_columns(df, inferred, {})
    recos = build_default_lever_recommendations(df, cols, {"ideaspace_retry_burst_count_trigger": 10, "ideaspace_retry_bucket_seconds": 60, "ideaspace_min_rows_for_reco": 200})
    assert any(r.lever_id == "retry_backoff" for r in recos)


def test_reco_affinity_triggered(run_dir: Path):
    df = pd.DataFrame({"host": (["h1"] * 200) + (["h2"] * 200), "duration": ([5.0] * 200) + ([10.0] * 200)})
    inferred = infer_columns(df, {"time_column": "auto", "group_by": "auto", "value_columns": "auto"})
    cols = pick_columns(df, inferred, {})
    recos = build_default_lever_recommendations(df, cols, {"ideaspace_affinity_multiplier_trigger": 0.9, "ideaspace_min_rows_for_reco": 200})
    assert any(r.lever_id == "resource_affinity" for r in recos)


def test_reco_parallelize_triggered(run_dir: Path):
    rows = []
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    for i in range(30):
        case = f"c{i}"
        if i % 2 == 0:
            rows.append((case, "A", base + pd.Timedelta(seconds=i * 10)))
            rows.append((case, "B", base + pd.Timedelta(seconds=i * 10 + 1)))
        else:
            rows.append((case, "B", base + pd.Timedelta(seconds=i * 10)))
            rows.append((case, "A", base + pd.Timedelta(seconds=i * 10 + 1)))
    # Keep ts SQLite-safe (strings).
    df = pd.DataFrame(rows, columns=["case_id", "activity", "ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True).astype(str)
    inferred = infer_columns(df, {"time_column": "auto", "group_by": "auto", "value_columns": "auto"})
    cols = pick_columns(df, inferred, {})
    recos = build_default_lever_recommendations(df, cols, {"ideaspace_parallelize_pair_min": 25, "ideaspace_parallelize_balance_trigger": 0.25})
    assert any(r.lever_id == "parallelize_branches" for r in recos)


def test_reco_prestage_triggered(run_dir: Path):
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    ts = [base] * 30 + [base + pd.Timedelta(seconds=3600 + i) for i in range(50)]
    activity = ["prereq"] * 30 + ["other"] * 50
    df = pd.DataFrame({"ts": pd.to_datetime(ts, utc=True).astype(str), "activity": activity})
    inferred = infer_columns(df, {"time_column": "auto", "group_by": "auto", "value_columns": "auto"})
    cols = pick_columns(df, inferred, {})
    recos = build_default_lever_recommendations(df, cols, {"ideaspace_prestage_min_count": 20, "ideaspace_bucket_seconds": 60})
    assert any(r.lever_id == "prestage_prereqs" for r in recos)
