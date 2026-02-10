from __future__ import annotations

from typing import Any

import csv

import pandas as pd

from statistic_harness.core.column_inference import infer_timestamp_series
from statistic_harness.core.dataset_io import DatasetAccessor
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.report_v2_utils import claim_id
from statistic_harness.core.utils import json_dumps, write_json


INVALID_STRINGS = {"", "nan", "none", "null"}
MAX_SAMPLE_ROWS = 50000
MIN_PROCESS_ALPHA_RATIO = 0.15
MIN_DATETIME_RATIO = 0.15


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _candidate_columns(
    preferred: str | None,
    columns: list[str],
    role_by_name: dict[str, str],
    roles: set[str],
    patterns: list[str],
    lower_names: dict[str, str],
    exclude: set[str],
) -> list[str]:
    seen: set[str] = set()
    candidates: list[str] = []
    if preferred and preferred in columns and preferred not in exclude and preferred not in seen:
        candidates.append(preferred)
        seen.add(preferred)
    for col in columns:
        if col in exclude or col in seen:
            continue
        if role_by_name.get(col) in roles:
            candidates.append(col)
            seen.add(col)
    for col in columns:
        if col in exclude or col in seen:
            continue
        name = lower_names[col]
        if any(pattern in name for pattern in patterns):
            candidates.append(col)
            seen.add(col)
    return candidates


def _sample_series(series: pd.Series, max_rows: int = MAX_SAMPLE_ROWS) -> pd.Series:
    if len(series) > max_rows:
        return series.head(max_rows)
    return series


def _normalize_values(series: pd.Series) -> pd.Series:
    sample = _sample_series(series)
    values = sample.dropna().astype(str).str.strip()
    if values.empty:
        return values
    values = values[~values.str.lower().isin(INVALID_STRINGS)]
    return values


def _score_process_column(series: pd.Series) -> tuple[float, float]:
    values = _normalize_values(series)
    if values.empty:
        return -1.0, 0.0
    alpha_ratio = float(values.str.contains(r"[A-Za-z]", regex=True).mean())
    digit_ratio = float(values.str.match(r"^\\d+(\\.0+)?$").mean())
    unique_ratio = float(values.nunique() / max(len(values), 1))
    score = alpha_ratio * 3.0 - digit_ratio * 2.0 + min(unique_ratio, 1.0) * 0.2
    return score, alpha_ratio


def _best_process_column(candidates: list[str], df: pd.DataFrame) -> str | None:
    best_col = None
    best_score = -1.0
    best_alpha = 0.0
    for col in candidates:
        score, alpha = _score_process_column(df[col])
        if score > best_score:
            best_col = col
            best_score = score
            best_alpha = alpha
    if best_col is None or best_alpha < MIN_PROCESS_ALPHA_RATIO:
        return None
    return best_col


def _best_datetime_column(
    candidates: list[str], df: pd.DataFrame, min_ratio: float = MIN_DATETIME_RATIO
) -> str | None:
    best_col = None
    best_score = 0.0
    for col in candidates:
        info = infer_timestamp_series(df[col], name_hint=col, sample_size=MAX_SAMPLE_ROWS)
        if not info.valid or info.parse_ratio < min_ratio:
            continue
        if info.score > best_score:
            best_score = info.score
            best_col = col
    return best_col


def _best_host_column(candidates: list[str], df: pd.DataFrame) -> str | None:
    best_col = None
    best_score = -1.0
    for col in candidates:
        values = _normalize_values(df[col])
        if values.empty:
            continue
        unique_ratio = float(values.nunique() / max(len(values), 1))
        alpha_ratio = float(values.str.contains(r"[A-Za-z]", regex=True).mean())
        score = unique_ratio + alpha_ratio * 0.2
        if score > best_score:
            best_score = score
            best_col = col
    return best_col


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def _top_n(entries: dict[str, float], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(entries.items(), key=lambda item: (-item[1], item[0]))
    return [
        {"id": key, "wait_sec": float(value)} for key, value in ordered[:limit]
    ]


class Plugin:
    def run(self, ctx) -> PluginResult:
        if not ctx.dataset_version_id:
            return PluginResult("skipped", "Dataset version missing", {}, [], [], None)

        accessor = DatasetAccessor(ctx.storage, ctx.dataset_version_id)
        df = accessor.load()
        if df.empty:
            return PluginResult("skipped", "Empty dataset", {}, [], [], None)

        columns_meta = []
        role_by_name: dict[str, str] = {}
        dataset_template = ctx.storage.fetch_dataset_template(ctx.dataset_version_id)
        if dataset_template and dataset_template.get("status") == "ready":
            fields = ctx.storage.fetch_template_fields(int(dataset_template["template_id"]))
            columns_meta = fields
            role_by_name = {field["name"]: (field.get("role") or "") for field in fields}
        else:
            columns_meta = ctx.storage.fetch_dataset_columns(ctx.dataset_version_id)
            role_by_name = {
                col["original_name"]: (col.get("role") or "")
                for col in columns_meta
            }

        columns = list(df.columns)
        lower_names = {col: str(col).lower() for col in columns}
        used: set[str] = set()

        process_candidates = _candidate_columns(
            ctx.settings.get("process_column"),
            columns,
            role_by_name,
            {"process", "activity", "event", "step", "task", "action"},
            ["process", "activity", "event", "step", "task", "action", "job"],
            lower_names,
            used,
        )
        process_col = _best_process_column(process_candidates, df)
        if process_col:
            used.add(process_col)

        host_candidates = _candidate_columns(
            ctx.settings.get("host_column"),
            columns,
            role_by_name,
            {"host", "machine", "server"},
            ["host", "machine", "server", "node"],
            lower_names,
            used,
        )
        host_col = _best_host_column(host_candidates, df)
        if host_col:
            used.add(host_col)

        queue_candidates = _candidate_columns(
            ctx.settings.get("queue_column"),
            columns,
            role_by_name,
            {"queue", "queued", "enqueue"},
            ["queue", "queued", "enqueue"],
            lower_names,
            used,
        )
        queue_col = _best_datetime_column(queue_candidates, df)
        if queue_col:
            used.add(queue_col)

        eligible_candidates = _candidate_columns(
            ctx.settings.get("eligible_column"),
            columns,
            role_by_name,
            {"eligible", "ready", "available"},
            ["eligible", "ready", "available"],
            lower_names,
            used,
        )
        eligible_col = _best_datetime_column(eligible_candidates, df)
        if eligible_col:
            used.add(eligible_col)

        start_candidates = _candidate_columns(
            ctx.settings.get("start_column"),
            columns,
            role_by_name,
            {"start"},
            ["start", "begin"],
            lower_names,
            used,
        )
        start_col = _best_datetime_column(start_candidates, df)
        if start_col:
            used.add(start_col)

        interval_start_col = eligible_col or queue_col
        if not interval_start_col or not start_col:
            return PluginResult(
                "skipped",
                "Missing eligible/queue or start timestamp columns",
                {},
                [],
                [],
                None,
            )

        work = df[[col for col in [process_col, host_col, interval_start_col, start_col] if col]].copy()
        work["__interval_start"] = _to_datetime(work[interval_start_col])
        work["__start_ts"] = _to_datetime(work[start_col])
        work = work.loc[work["__interval_start"].notna() & work["__start_ts"].notna()].copy()
        if work.empty:
            return PluginResult("ok", "No valid timestamps for busy period analysis", {}, [], [], None)

        wait_to_start = (work["__start_ts"] - work["__interval_start"]).dt.total_seconds()
        wait_to_start = wait_to_start.clip(lower=0).fillna(0.0)
        threshold = float(ctx.settings.get("wait_threshold_seconds", 60))
        work["__over_threshold_sec"] = (wait_to_start - threshold).clip(lower=0).fillna(0.0)

        work = work.loc[work["__over_threshold_sec"] > 0].copy()
        if work.empty:
            metrics = {
                "busy_periods": 0,
                "runs_over_threshold": 0,
                "wait_threshold_seconds": threshold,
                "gap_tolerance_seconds": float(ctx.settings.get("gap_tolerance_seconds", 60)),
            }
            return PluginResult("ok", "No over-threshold runs", metrics, [], [], None)

        if process_col:
            work["__process"] = work[process_col].map(_normalize_text)
        else:
            work["__process"] = ""
        if host_col:
            work["__host"] = work[host_col].map(_normalize_text)
        else:
            work["__host"] = ""

        work["__interval_end"] = work["__interval_start"] + pd.to_timedelta(
            work["__over_threshold_sec"], unit="s"
        )

        intervals = list(
            zip(
                work["__interval_start"],
                work["__interval_end"],
                work["__over_threshold_sec"],
                work["__process"],
                work["__host"],
            )
        )
        intervals.sort(key=lambda row: (row[0], row[1]))

        gap_tolerance = float(ctx.settings.get("gap_tolerance_seconds", 60))
        busy_periods: list[dict[str, Any]] = []
        current = None
        process_sums: dict[str, float] = {}
        host_sums: dict[str, float] = {}
        run_count = 0

        def _flush() -> None:
            nonlocal current, process_sums, host_sums, run_count
            if not current:
                return
            start_ts, end_ts, total_sec = current
            duration_sec = (end_ts - start_ts).total_seconds()
            top_process = _top_n(process_sums, 1)
            top_host = _top_n(host_sums, 1)
            period_id = f"bp_{len(busy_periods) + 1:04d}"
            busy_periods.append(
                {
                    "busy_period_id": period_id,
                    "start_ts": start_ts.isoformat(),
                    "end_ts": end_ts.isoformat(),
                    "duration_sec": float(duration_sec),
                    "total_over_threshold_wait_sec": float(total_sec),
                    "runs_over_threshold_count": int(run_count),
                    "top_process_by_wait": top_process[0] if top_process else None,
                    "top_host_by_wait": top_host[0] if top_host else None,
                    "per_process_over_threshold_wait_sec": {
                        key: float(value) for key, value in process_sums.items()
                    },
                    "per_host_over_threshold_wait_sec": {
                        key: float(value) for key, value in host_sums.items()
                    },
                    "weekday": start_ts.day_name(),
                    "weekend": bool(start_ts.dayofweek >= 5),
                    "after_hours": bool(start_ts.hour < 8 or start_ts.hour >= 18),
                }
            )
            current = None
            process_sums = {}
            host_sums = {}
            run_count = 0

        for start_ts, end_ts, over_sec, process, host in intervals:
            if current is None:
                current = [start_ts, end_ts, float(over_sec)]
            else:
                gap = (start_ts - current[1]).total_seconds()
                if gap <= gap_tolerance:
                    if end_ts > current[1]:
                        current[1] = end_ts
                    current[2] += float(over_sec)
                else:
                    _flush()
                    current = [start_ts, end_ts, float(over_sec)]
            run_count += 1
            if process:
                process_sums[process] = process_sums.get(process, 0.0) + float(over_sec)
            if host:
                host_sums[host] = host_sums.get(host, 0.0) + float(over_sec)

        _flush()

        busy_periods = sorted(
            busy_periods,
            key=lambda row: (
                -float(row.get("total_over_threshold_wait_sec") or 0.0),
                row.get("start_ts") or "",
                row.get("busy_period_id") or "",
            ),
        )

        artifacts_dir = ctx.artifacts_dir("analysis_busy_period_segmentation_v2")
        definition_path = artifacts_dir / "definition.json"
        write_json(
            definition_path,
            {
                "wait_threshold_seconds": threshold,
                "gap_tolerance_seconds": gap_tolerance,
                "interval_start_column": interval_start_col,
                "start_column": start_col,
                "process_column": process_col,
                "host_column": host_col,
            },
        )
        full_path = artifacts_dir / "busy_periods.json"
        write_json(full_path, {"busy_periods": busy_periods})

        top_n = int(ctx.settings.get("max_top_rows", 50))
        process_top_n = int(ctx.settings.get("max_top_processes", 10))
        host_top_n = int(ctx.settings.get("max_top_hosts", 10))
        csv_rows: list[dict[str, Any]] = []
        for row in busy_periods[:top_n]:
            per_process = row.get("per_process_over_threshold_wait_sec") or {}
            per_host = row.get("per_host_over_threshold_wait_sec") or {}
            csv_rows.append(
                {
                    "busy_period_id": row.get("busy_period_id"),
                    "start_ts": row.get("start_ts"),
                    "end_ts": row.get("end_ts"),
                    "duration_sec": row.get("duration_sec"),
                    "total_over_threshold_wait_sec": row.get(
                        "total_over_threshold_wait_sec"
                    ),
                    "total_over_threshold_wait_hours": round(
                        float(row.get("total_over_threshold_wait_sec") or 0.0) / 3600.0,
                        2,
                    ),
                    "runs_over_threshold_count": row.get("runs_over_threshold_count"),
                    "weekday": row.get("weekday"),
                    "weekend": row.get("weekend"),
                    "after_hours": row.get("after_hours"),
                    "top_process_id": (row.get("top_process_by_wait") or {}).get("id"),
                    "top_process_wait_sec": (row.get("top_process_by_wait") or {}).get("wait_sec"),
                    "top_host_id": (row.get("top_host_by_wait") or {}).get("id"),
                    "top_host_wait_sec": (row.get("top_host_by_wait") or {}).get("wait_sec"),
                    "claim_id": claim_id(
                        f"busy_period:{row.get('busy_period_id')}:total"
                    ),
                    "process_top10": json_dumps(_top_n(per_process, process_top_n)),
                    "host_top10": json_dumps(_top_n(per_host, host_top_n)),
                }
            )

        slide_path = ctx.run_dir / "slide_kit" / "busy_periods.csv"
        slide_path.parent.mkdir(parents=True, exist_ok=True)
        if csv_rows:
            headers = list(csv_rows[0].keys())
        else:
            headers = [
                "busy_period_id",
                "start_ts",
                "end_ts",
                "duration_sec",
                "total_over_threshold_wait_sec",
                "total_over_threshold_wait_hours",
                "runs_over_threshold_count",
                "weekday",
                "weekend",
                "after_hours",
                "top_process_id",
                "top_process_wait_sec",
                "top_host_id",
                "top_host_wait_sec",
                "claim_id",
                "process_top10",
                "host_top10",
            ]
        with slide_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow({key: row.get(key, "") for key in headers})

        artifacts = [
            PluginArtifact(
                path=str(full_path.relative_to(ctx.run_dir)),
                type="json",
                description="Busy period segmentation (full)",
            ),
            PluginArtifact(
                path=str(definition_path.relative_to(ctx.run_dir)),
                type="json",
                description="Busy period segmentation definition",
            ),
            PluginArtifact(
                path=str(slide_path.relative_to(ctx.run_dir)),
                type="csv",
                description="Busy period segmentation (top rows)",
            ),
        ]

        total_wait_sec = sum(
            float(row.get("total_over_threshold_wait_sec") or 0.0)
            for row in busy_periods
        )
        findings = [
            {
                "kind": "busy_period_summary",
                "busy_periods": len(busy_periods),
                "total_over_threshold_wait_hours": total_wait_sec / 3600.0,
                "measurement_type": "measured",
            }
        ]

        metrics = {
            "busy_periods": len(busy_periods),
            "runs_over_threshold": int(work.shape[0]),
            "wait_threshold_seconds": threshold,
            "gap_tolerance_seconds": gap_tolerance,
        }

        return PluginResult(
            "ok",
            f"Computed {len(busy_periods)} busy periods",
            metrics,
            findings,
            artifacts,
            None,
        )
