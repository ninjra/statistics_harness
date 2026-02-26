from __future__ import annotations

from pathlib import Path
from typing import Any

import json

import numpy as np
import pandas as pd

from statistic_harness.core.utils import stable_hash


DEFAULT_PAYOUT_PROCESS_REGEX = r"(?i)(^|_)(pay|payout)|jbprepay|adppaystat"


def _pick_process_col(df: pd.DataFrame) -> str | None:
    for name in ("PROCESS_ID", "process_id", "process_norm", "activity", "ACTIVITY"):
        if name in df.columns:
            return name
    for col in df.columns:
        if "process" in str(col).lower():
            return str(col)
    return None


def _pick_param_col(df: pd.DataFrame) -> str | None:
    for name in ("PARAM_DESCR_LIST", "param_descr_list", "PARAMS", "params"):
        if name in df.columns:
            return name
    for col in df.columns:
        lowered = str(col).lower()
        if any(tok in lowered for tok in ("pay_period", "period", "batch", "pay", "payout", "run_key", "key")):
            return str(col)
    return None


def _pick_time_cols(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    queue = None
    start = None
    end = None
    for col in df.columns:
        c = str(col).lower()
        if queue is None and "queue" in c:
            queue = str(col)
        if start is None and "start" in c:
            start = str(col)
        if end is None and ("end" in c or "finish" in c or "completed" in c):
            end = str(col)
    return queue, start, end


def _hash_token(value: str) -> str:
    return f"tok_{stable_hash(value):08x}"


def _safe_to_datetime_series(series: pd.Series) -> pd.Series:
    # Convert via object ndarray with cache disabled to avoid Arrow unique-cache
    # allocations on large datasets.
    parsed = pd.to_datetime(series.to_numpy(dtype=object, copy=False), errors="coerce", cache=False)
    return pd.Series(parsed, index=series.index)


def build_payout_report(
    df: pd.DataFrame,
    *,
    payout_process_regex: str = DEFAULT_PAYOUT_PROCESS_REGEX,
) -> dict[str, Any]:
    proc_col = _pick_process_col(df)
    if not proc_col:
        return {
            "schema_version": "v1",
            "status": "skipped",
            "summary": "No process column found for payout report",
            "metrics": {},
            "per_source": [],
        }

    source_col = "__source_file" if "__source_file" in df.columns else None
    param_col = _pick_param_col(df)
    queue_col, start_col, end_col = _pick_time_cols(df)

    proc = df[proc_col].astype(str)
    payout_mask = proc.str.contains(payout_process_regex, regex=True, na=False)

    duration_s = None
    queue_delay_s = None
    if start_col and end_col and start_col in df.columns and end_col in df.columns:
        s = _safe_to_datetime_series(df[start_col])
        e = _safe_to_datetime_series(df[end_col])
        ok = s.notna() & e.notna()
        if ok.any():
            duration_s = (e[ok] - s[ok]).dt.total_seconds().clip(lower=0.0)
    if queue_col and start_col and queue_col in df.columns and start_col in df.columns:
        q = _safe_to_datetime_series(df[queue_col])
        s = _safe_to_datetime_series(df[start_col])
        ok = q.notna() & s.notna()
        if ok.any():
            queue_delay_s = (s[ok] - q[ok]).dt.total_seconds().clip(lower=0.0)

    def summarize(frame: pd.DataFrame) -> dict[str, Any]:
        total_rows = int(len(frame))
        payout_rows = int(payout_mask.loc[frame.index].sum())
        top_procs = (
            proc.loc[frame.index][payout_mask.loc[frame.index]]
            .value_counts()
            .head(10)
            .to_dict()
        )
        out: dict[str, Any] = {
            "total_rows": total_rows,
            "payout_rows": payout_rows,
            "top_payout_processes": {str(k): int(v) for k, v in top_procs.items()},
        }
        if duration_s is not None:
            idx = frame.index.intersection(duration_s.index)
            ds = duration_s.loc[idx]
            out["payout_duration_hours"] = float(np.nansum(ds.to_numpy(dtype=float)) / 3600.0)
        if queue_delay_s is not None:
            idx = frame.index.intersection(queue_delay_s.index)
            qs = queue_delay_s.loc[idx]
            out["payout_queue_delay_hours"] = float(np.nansum(qs.to_numpy(dtype=float)) / 3600.0)
        if param_col and param_col in frame.columns:
            params = frame.loc[payout_mask.loc[frame.index], param_col].astype(str)
            values = params.to_numpy(dtype=str, copy=False)
            out["unique_payout_keys"] = int(len({_hash_token(v) for v in values if v}))
        return out

    per_source: list[dict[str, Any]] = []
    if source_col:
        for source, group in df.groupby(source_col, dropna=False):
            row = {"source": str(source)}
            row.update(summarize(group))
            per_source.append(row)
        per_source.sort(key=lambda r: (-int(r.get("payout_rows") or 0), str(r.get("source") or "")))

    overall = summarize(df)
    return {
        "schema_version": "v1",
        "status": "ok",
        "summary": "Built payout report",
        "process_col": proc_col,
        "param_col": param_col or "",
        "source_col": source_col or "",
        "metrics": overall,
        "per_source": per_source,
        "privacy": {
            "param_keys": "hashed",
            "pii": "not emitted",
        },
    }


def write_payout_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "payout_report.json"
    csv_path = out_dir / "payout_report.csv"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    overall = dict(report.get("metrics") or {})
    overall_row = {"scope": "overall", **overall}
    rows.append(overall_row)
    for row in report.get("per_source") or []:
        if isinstance(row, dict):
            rows.append({"scope": "source", **row})
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, json_path
