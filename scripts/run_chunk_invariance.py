#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from statistic_harness.core.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _load_tolerances(path: Path) -> dict[str, float]:
    defaults = {"hours_abs": 0.01, "percent_abs": 0.001, "count_abs": 0.0}
    if not path.exists():
        return defaults
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return defaults
    if not isinstance(payload, dict):
        return defaults
    out = dict(defaults)
    for key in list(out.keys()):
        try:
            out[key] = float(payload.get(key, out[key]))
        except (TypeError, ValueError):
            continue
    return out


def _report_signature(report: dict[str, Any]) -> dict[str, Any]:
    rec_block = report.get("recommendations") if isinstance(report.get("recommendations"), dict) else {}
    items = rec_block.get("items") if isinstance(rec_block.get("items"), list) else []
    typed_items = [row for row in items if isinstance(row, dict)]
    total_delta_hours = 0.0
    total_delta_close_dynamic = 0.0
    avg_eff_pct_close_dynamic_vals: list[float] = []
    for row in typed_items:
        d = _as_float(row.get("modeled_delta_hours"))
        if isinstance(d, float) and d > 0.0:
            total_delta_hours += d
        dd = _as_float(
            row.get("modeled_delta_hours_close_cycle")
            if row.get("modeled_delta_hours_close_cycle") is not None
            else row.get("delta_hours_close_dynamic")
        )
        if isinstance(dd, float) and dd > 0.0:
            total_delta_close_dynamic += dd
        pct = _as_float(
            row.get("modeled_efficiency_gain_pct_close_cycle")
            if row.get("modeled_efficiency_gain_pct_close_cycle") is not None
            else row.get("efficiency_gain_pct_close_dynamic")
        )
        if isinstance(pct, float) and pct >= 0.0:
            avg_eff_pct_close_dynamic_vals.append(pct)
    plugins = report.get("plugins") if isinstance(report.get("plugins"), dict) else {}
    status_counts: dict[str, int] = {}
    for payload in plugins.values():
        if not isinstance(payload, dict):
            continue
        status = str(payload.get("status") or "").strip().lower() or "unknown"
        status_counts[status] = int(status_counts.get(status, 0)) + 1
    return {
        "recommendation_count": len(typed_items),
        "total_modeled_delta_hours": round(total_delta_hours, 6),
        "total_modeled_delta_hours_close_cycle": round(total_delta_close_dynamic, 6),
        "avg_efficiency_gain_pct_close_cycle": round(
            sum(avg_eff_pct_close_dynamic_vals) / len(avg_eff_pct_close_dynamic_vals), 6
        )
        if avg_eff_pct_close_dynamic_vals
        else None,
        "status_counts": status_counts,
    }


def _compare_signature(
    baseline: dict[str, Any], candidate: dict[str, Any], tolerances: dict[str, float]
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    count_tol = int(tolerances.get("count_abs", 0.0))
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


def _execute_run(
    *,
    appdata_root: Path,
    input_file: Path,
    plugin_ids: list[str],
    run_seed: int,
    batch_size: int,
) -> str:
    appdata_root.mkdir(parents=True, exist_ok=True)
    old_appdata = os.environ.get("STAT_HARNESS_APPDATA")
    old_force_batch = os.environ.get("STAT_HARNESS_FORCE_BATCH_SIZE")
    try:
        os.environ["STAT_HARNESS_APPDATA"] = str(appdata_root)
        os.environ["STAT_HARNESS_FORCE_BATCH_SIZE"] = str(int(batch_size))
        pipeline = Pipeline(appdata_root, ROOT / "plugins")
        return pipeline.run(input_file, plugin_ids, {}, int(run_seed))
    finally:
        if old_appdata is None:
            os.environ.pop("STAT_HARNESS_APPDATA", None)
        else:
            os.environ["STAT_HARNESS_APPDATA"] = old_appdata
        if old_force_batch is None:
            os.environ.pop("STAT_HARNESS_FORCE_BATCH_SIZE", None)
        else:
            os.environ["STAT_HARNESS_FORCE_BATCH_SIZE"] = old_force_batch


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or compare chunk-size invariance signatures.")
    parser.add_argument("--run-ids", default="")
    parser.add_argument("--input-file", default="")
    parser.add_argument("--plugin-ids", default="")
    parser.add_argument("--batch-sizes", default="1000,10000,50000")
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--appdata-root", default=str(ROOT / "appdata" / "chunk_invariance"))
    parser.add_argument(
        "--tolerances",
        default=str(ROOT / "config" / "chunk_invariance_tolerances.yaml"),
    )
    parser.add_argument("--out-json", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    run_ids = [token.strip() for token in str(args.run_ids).split(",") if token.strip()]
    if not run_ids:
        input_file = Path(str(args.input_file).strip())
        if not input_file.is_absolute():
            input_file = ROOT / input_file
        plugin_ids = [token.strip() for token in str(args.plugin_ids).split(",") if token.strip()]
        if not plugin_ids:
            raise SystemExit("plugin ids required when run-ids not provided")
        batch_sizes = [int(token.strip()) for token in str(args.batch_sizes).split(",") if token.strip()]
        if not batch_sizes:
            raise SystemExit("at least one batch size is required")
        appdata_root = Path(str(args.appdata_root))
        generated: list[str] = []
        for batch_size in batch_sizes:
            run_id = _execute_run(
                appdata_root=appdata_root / f"batch_{batch_size}",
                input_file=input_file,
                plugin_ids=plugin_ids,
                run_seed=int(args.run_seed),
                batch_size=int(batch_size),
            )
            generated.append(run_id)
        run_ids = generated

    runs_root = ROOT / "appdata" / "runs"
    signatures: dict[str, dict[str, Any]] = {}
    for run_id in run_ids:
        report = _load_json(runs_root / run_id / "report.json")
        signatures[run_id] = _report_signature(report)

    baseline_id = run_ids[0]
    baseline = signatures[baseline_id]
    tolerances = _load_tolerances(Path(str(args.tolerances)))
    comparisons: list[dict[str, Any]] = []
    overall_ok = True
    for run_id in run_ids[1:]:
        ok, errors = _compare_signature(baseline, signatures[run_id], tolerances)
        if not ok:
            overall_ok = False
        comparisons.append(
            {
                "baseline_run_id": baseline_id,
                "candidate_run_id": run_id,
                "ok": ok,
                "errors": errors,
            }
        )

    payload = {
        "schema_version": "chunk_invariance.v1",
        "ok": overall_ok,
        "baseline_run_id": baseline_id,
        "run_ids": run_ids,
        "tolerances": tolerances,
        "signatures": signatures,
        "comparisons": comparisons,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out_json = str(args.out_json).strip()
    if out_json:
        out_path = Path(out_json)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if bool(args.strict) and not overall_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

