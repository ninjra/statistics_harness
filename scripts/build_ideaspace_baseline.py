from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from statistic_harness.core.baselines import schema_sha256, signed_digest
from statistic_harness.core.ideaspace_feature_extractor import (
    kpi_summary,
    pick_columns,
    time_span_seconds,
    duration_seconds,
    queue_delay_seconds,
)
from statistic_harness.core.stat_plugins.columns import infer_columns
from statistic_harness.core.utils import now_iso, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline-id", default="quorum_baseline_v1")
    parser.add_argument("--version", default="1.0.0")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    inferred = infer_columns(df, {"time_column": "auto", "group_by": "auto", "value_columns": "auto"})
    cols = pick_columns(df, inferred, {"ideaspace_max_group_cols": 0})

    ideal: dict[str, float] = {}
    span = time_span_seconds(df, cols.time_col)
    if span is not None:
        ideal["time_span_s"] = float(span)
        ideal["rate_per_min"] = float(len(df) / max(span / 60.0, 1e-9))
    dur = duration_seconds(df, cols.duration_col, cols.start_col, cols.end_col)
    dur_kpi = kpi_summary(dur)
    if dur_kpi:
        ideal["duration_p50"] = float(dur_kpi["p50"])
        ideal["duration_p95"] = float(dur_kpi["p95"])
    qd = queue_delay_seconds(df, cols.eligible_col, cols.start_col)
    qd_kpi = kpi_summary(qd)
    if qd_kpi:
        ideal["queue_delay_p50"] = float(qd_kpi["p50"])
        ideal["queue_delay_p95"] = float(qd_kpi["p95"])

    schema_path = ROOT / "docs" / "ideaspace_baseline.schema.json"
    baseline = {
        "baseline_id": str(args.baseline_id),
        "version": str(args.version),
        "created_at": now_iso(),
        "schema_hash": schema_sha256(schema_path),
        "ideal_vector": ideal,
        "signature": {"algo": "sha256", "digest": ""},
    }
    baseline["signature"]["digest"] = signed_digest(baseline)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

