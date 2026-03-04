"""Four-pillars contract: chunk-size invariance.

Runs a small set of plugins at different batch sizes and asserts that the
report signature stays within the tolerances defined in
``config/chunk_invariance_tolerances.yaml``.

Acceptance criteria (Task 2.2):
  - hours_abs  <= 0.01
  - percent_abs <= 0.001
  - count_abs  == 0
  - Plugin status counts must be identical across chunk sizes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from statistic_harness.core.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
TOLERANCES_PATH = ROOT / "config" / "chunk_invariance_tolerances.yaml"

# Lightweight plugins that are fast and deterministic — enough to produce a
# report signature with recommendations.
INVARIANCE_PLUGINS = [
    "profile_basic",
    "analysis_percentile_analysis",
    "analysis_tail_isolation",
]

BATCH_SIZES = [500, 1000, 5000]


def _load_tolerances() -> dict[str, float]:
    defaults = {"hours_abs": 0.01, "percent_abs": 0.001, "count_abs": 0.0}
    if not TOLERANCES_PATH.exists():
        return defaults
    payload = yaml.safe_load(TOLERANCES_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return defaults
    for key in list(defaults.keys()):
        try:
            defaults[key] = float(payload.get(key, defaults[key]))
        except (TypeError, ValueError):
            continue
    return defaults


def _as_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _report_signature(report: dict) -> dict:
    rec_block = report.get("recommendations") if isinstance(report.get("recommendations"), dict) else {}
    items = rec_block.get("items") if isinstance(rec_block.get("items"), list) else []
    typed_items = [row for row in items if isinstance(row, dict)]

    total_delta_hours = 0.0
    total_delta_close_dynamic = 0.0
    avg_vals: list[float] = []
    for row in typed_items:
        d = _as_float(row.get("modeled_delta_hours"))
        if isinstance(d, float) and d > 0.0:
            total_delta_hours += d
        dd = _as_float(
            row.get("modeled_delta_hours_close_cycle")
            or row.get("delta_hours_close_dynamic")
        )
        if isinstance(dd, float) and dd > 0.0:
            total_delta_close_dynamic += dd
        pct = _as_float(
            row.get("modeled_efficiency_gain_pct_close_cycle")
            or row.get("efficiency_gain_pct_close_dynamic")
        )
        if isinstance(pct, float) and pct >= 0.0:
            avg_vals.append(pct)

    plugins = report.get("plugins") if isinstance(report.get("plugins"), dict) else {}
    status_counts: dict[str, int] = {}
    for payload in plugins.values():
        if not isinstance(payload, dict):
            continue
        status = str(payload.get("status") or "").strip().lower() or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "recommendation_count": len(typed_items),
        "total_modeled_delta_hours": round(total_delta_hours, 6),
        "total_modeled_delta_hours_close_cycle": round(total_delta_close_dynamic, 6),
        "avg_efficiency_gain_pct_close_cycle": round(
            sum(avg_vals) / len(avg_vals), 6
        )
        if avg_vals
        else None,
        "status_counts": status_counts,
    }


def _compare_signatures(
    baseline: dict, candidate: dict, tolerances: dict[str, float]
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    count_tol = int(tolerances.get("count_abs", 0))
    hours_tol = float(tolerances.get("hours_abs", 0.01))
    pct_tol = float(tolerances.get("percent_abs", 0.001))

    if abs(int(candidate["recommendation_count"]) - int(baseline["recommendation_count"])) > count_tol:
        errors.append("RECOMMENDATION_COUNT_DRIFT")
    if abs(float(candidate["total_modeled_delta_hours"]) - float(baseline["total_modeled_delta_hours"])) > hours_tol:
        errors.append("TOTAL_MODELED_DELTA_HOURS_DRIFT")
    if (
        abs(
            float(candidate["total_modeled_delta_hours_close_cycle"])
            - float(baseline["total_modeled_delta_hours_close_cycle"])
        )
        > hours_tol
    ):
        errors.append("CLOSE_DYNAMIC_DELTA_HOURS_DRIFT")

    base_pct = baseline.get("avg_efficiency_gain_pct_close_cycle")
    cand_pct = candidate.get("avg_efficiency_gain_pct_close_cycle")
    if isinstance(base_pct, (int, float)) and isinstance(cand_pct, (int, float)):
        if abs(float(cand_pct) - float(base_pct)) > pct_tol:
            errors.append("CLOSE_DYNAMIC_EFFICIENCY_PCT_DRIFT")

    if dict(candidate.get("status_counts") or {}) != dict(baseline.get("status_counts") or {}):
        errors.append("PLUGIN_STATUS_COUNTS_DRIFT")

    return (len(errors) == 0), errors


def _run_pipeline(tmp_path: Path, batch_size: int) -> dict:
    appdata = tmp_path / f"batch_{batch_size}"
    appdata.mkdir(parents=True, exist_ok=True)
    old_appdata = os.environ.get("STAT_HARNESS_APPDATA")
    old_batch = os.environ.get("STAT_HARNESS_FORCE_BATCH_SIZE")
    try:
        os.environ["STAT_HARNESS_APPDATA"] = str(appdata)
        os.environ["STAT_HARNESS_FORCE_BATCH_SIZE"] = str(batch_size)
        fixture = ROOT / "tests" / "fixtures" / "synth_linear.csv"
        pipeline = Pipeline(appdata, ROOT / "plugins")
        run_id = pipeline.run(fixture, INVARIANCE_PLUGINS, {}, 42)
        report_path = appdata / "runs" / run_id / "report.json"
        return json.loads(report_path.read_text(encoding="utf-8"))
    finally:
        if old_appdata is None:
            os.environ.pop("STAT_HARNESS_APPDATA", None)
        else:
            os.environ["STAT_HARNESS_APPDATA"] = old_appdata
        if old_batch is None:
            os.environ.pop("STAT_HARNESS_FORCE_BATCH_SIZE", None)
        else:
            os.environ["STAT_HARNESS_FORCE_BATCH_SIZE"] = old_batch


@pytest.mark.slow
def test_chunk_invariance_across_batch_sizes(tmp_path: Path) -> None:
    """Report signatures must match within config tolerances across batch sizes."""
    tolerances = _load_tolerances()
    signatures: dict[int, dict] = {}
    for batch_size in BATCH_SIZES:
        report = _run_pipeline(tmp_path, batch_size)
        signatures[batch_size] = _report_signature(report)

    baseline = signatures[BATCH_SIZES[0]]
    for batch_size in BATCH_SIZES[1:]:
        ok, errors = _compare_signatures(baseline, signatures[batch_size], tolerances)
        assert ok, (
            f"Chunk invariance violation at batch_size={batch_size}: {errors}\n"
            f"  baseline({BATCH_SIZES[0]}): {baseline}\n"
            f"  candidate({batch_size}): {signatures[batch_size]}"
        )


@pytest.mark.slow
def test_chunk_invariance_status_counts_exact(tmp_path: Path) -> None:
    """Plugin status counts must be identical regardless of chunk size."""
    signatures: dict[int, dict] = {}
    for batch_size in BATCH_SIZES:
        report = _run_pipeline(tmp_path, batch_size)
        signatures[batch_size] = _report_signature(report)

    baseline_counts = signatures[BATCH_SIZES[0]]["status_counts"]
    for batch_size in BATCH_SIZES[1:]:
        assert signatures[batch_size]["status_counts"] == baseline_counts, (
            f"Status count drift at batch_size={batch_size}: "
            f"expected {baseline_counts}, "
            f"got {signatures[batch_size]['status_counts']}"
        )


def test_tolerances_config_loads() -> None:
    """Config YAML must load and contain expected keys."""
    tolerances = _load_tolerances()
    assert "hours_abs" in tolerances
    assert "percent_abs" in tolerances
    assert "count_abs" in tolerances
    assert tolerances["hours_abs"] <= 0.01
    assert tolerances["percent_abs"] <= 0.001
    assert tolerances["count_abs"] == 0.0
