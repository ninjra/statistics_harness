from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from statistic_harness.core.column_inference import infer_timestamp_series
from statistic_harness.core.accounting_windows import parse_accounting_month_from_params
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json

PARAM_PATTERN = re.compile(r"\s*([^;=]+?)\s*\([^\)]*\)\s*=\s*([^;]+)")


@dataclass(frozen=True)
class CloseWindow:
    accounting_month: str
    roll_ts: datetime
    close_start: datetime
    close_end: datetime
    window_days: float
    indicator_processes: list[dict[str, Any]]


def _candidate_columns(
    preferred: str | None,
    columns: list[str],
    role_by_name: dict[str, str],
    roles: set[str],
    patterns: list[str],
    lower_names: dict[str, str],
) -> list[str]:
    seen: set[str] = set()
    candidates: list[str] = []
    if preferred and preferred in columns:
        candidates.append(preferred)
        seen.add(preferred)
    for col in columns:
        if role_by_name.get(col) in roles and col not in seen:
            candidates.append(col)
            seen.add(col)
    for col in columns:
        if col in seen:
            continue
        name = lower_names[col]
        if any(pattern in name for pattern in patterns):
            candidates.append(col)
            seen.add(col)
    return candidates


def _best_datetime_column(candidates: list[str], df: pd.DataFrame) -> str | None:
    best_col = None
    best_score = 0.0
    for col in candidates:
        info = infer_timestamp_series(df[col], name_hint=col, sample_size=2000)
        if not info.valid:
            continue
        if info.score > best_score:
            best_score = info.score
            best_col = col
    return best_col


def _parse_params(text: str) -> dict[str, str]:
    if not text:
        return {}
    out: dict[str, str] = {}
    for match in PARAM_PATTERN.finditer(text):
        key = match.group(1).strip().lower()
        value = match.group(2).strip()
        if key:
            out[key] = value
    return out


def _parse_accounting_month(
    params: dict[str, str], *, reference_ts: datetime | None = None
) -> datetime | None:
    return parse_accounting_month_from_params(params, reference_ts=reference_ts)


def _previous_month_start(dt: datetime) -> datetime:
    year = dt.year
    month = dt.month - 1
    if month == 0:
        month = 12
        year -= 1
    return datetime(year, month, 1)


def _close_start_from_roll(roll_month: datetime, day: int) -> datetime:
    prev_month = _previous_month_start(roll_month)
    return datetime(prev_month.year, prev_month.month, day)


class Plugin:
    def run(self, ctx) -> PluginResult:
        df = ctx.dataset_loader()
        if df.empty:
            return PluginResult("skipped", "Empty dataset", {}, [], [], None)

        columns_meta = []
        role_by_name: dict[str, str] = {}
        if ctx.dataset_version_id:
            dataset_template = ctx.storage.fetch_dataset_template(ctx.dataset_version_id)
            if dataset_template and dataset_template.get("status") == "ready":
                fields = ctx.storage.fetch_template_fields(
                    int(dataset_template["template_id"])
                )
                columns_meta = fields
                role_by_name = {
                    field["name"]: (field.get("role") or "") for field in fields
                }
            else:
                columns_meta = ctx.storage.fetch_dataset_columns(ctx.dataset_version_id)
                role_by_name = {
                    col["original_name"]: (col.get("role") or "")
                    for col in columns_meta
                }

        columns = list(df.columns)
        lower_names = {col: str(col).lower() for col in columns}

        if "PROCESS_ID" in columns:
            process_col = "PROCESS_ID"
        else:
            process_candidates = _candidate_columns(
                ctx.settings.get("process_column"),
                columns,
                role_by_name,
                {"process", "activity", "event", "step", "task"},
                ["process", "activity", "event", "step", "task", "job"],
                lower_names,
            )
            process_col = process_candidates[0] if process_candidates else None

        if "PARAM_DESCR_LIST" in columns:
            params_col = "PARAM_DESCR_LIST"
        else:
            params_candidates = _candidate_columns(
                ctx.settings.get("params_column"),
                columns,
                role_by_name,
                {"params", "attributes"},
                ["param", "descr", "argument", "input"],
                lower_names,
            )
            params_col = params_candidates[0] if params_candidates else None

        if "START_DT" in columns:
            start_col = "START_DT"
        else:
            start_candidates = _candidate_columns(
                ctx.settings.get("start_column"),
                columns,
                role_by_name,
                {"start_time", "start"},
                ["start", "begin"],
                lower_names,
            )
            start_col = _best_datetime_column(start_candidates, df) if start_candidates else None

        if "QUEUE_DT" in columns:
            queue_col = "QUEUE_DT"
        else:
            queue_candidates = _candidate_columns(
                ctx.settings.get("queue_column"),
                columns,
                role_by_name,
                {"queue_time", "queue"},
                ["queue", "enqueue", "submitted"],
                lower_names,
            )
            queue_col = _best_datetime_column(queue_candidates, df) if queue_candidates else None

        if not process_col or not params_col or not (start_col or queue_col):
            return PluginResult(
                "ok",
                "Dynamic close detection not applicable",
                {},
                [],
                [],
                None,
            )

        process_series = df[process_col].astype(str)
        params_series = df[params_col].fillna("").astype(str)
        start_ts = pd.to_datetime(df[start_col], errors="coerce") if start_col else None
        queue_ts = pd.to_datetime(df[queue_col], errors="coerce") if queue_col else None
        ts = start_ts.fillna(queue_ts) if start_ts is not None else queue_ts

        records: list[dict[str, Any]] = []
        for idx, params_text in enumerate(params_series):
            stamp = ts.iloc[idx] if ts is not None else pd.NaT
            if pd.isna(stamp):
                continue
            params = _parse_params(params_text)
            acct_month = _parse_accounting_month(
                params, reference_ts=stamp.to_pydatetime()
            )
            if not acct_month:
                continue
            records.append(
                {
                    "process": process_series.iloc[idx].strip().lower(),
                    "acct_month": acct_month,
                    "ts": stamp.to_pydatetime(),
                }
            )

        if not records:
            return PluginResult(
                "ok",
                "No accounting month markers found for dynamic close detection.",
                {"months_detected": 0},
                [],
                [],
                None,
            )

        window_hours = float(ctx.settings.get("indicator_window_hours") or 24)
        close_start_day = int(ctx.settings.get("close_start_day") or 20)
        min_indicator_months = int(ctx.settings.get("indicator_min_months") or 3)
        dominance_min_days = int(ctx.settings.get("dominance_min_days") or 3)
        dominance_min_share = float(ctx.settings.get("dominance_min_share") or 0.5)

        rec_df = pd.DataFrame(records)
        rec_df["month_key"] = rec_df["acct_month"].dt.strftime("%Y-%m")
        rec_df["day"] = rec_df["ts"].dt.date
        day_counts = (
            rec_df.groupby(["day", "month_key"]).size().reset_index(name="count")
        )
        totals = day_counts.groupby("day")["count"].sum().reset_index(name="total")
        merged = day_counts.merge(totals, on="day", how="left")
        merged["share"] = merged["count"] / merged["total"].clip(lower=1)

        dominant = (
            merged.sort_values(["day", "count"], ascending=[True, False])
            .groupby("day")
            .head(1)
            .reset_index(drop=True)
        )

        min_ts_map = (
            rec_df.groupby(["day", "month_key"])['ts']
            .min()
            .to_dict()
        )

        close_windows: list[CloseWindow] = []
        indicator_stats: dict[str, dict[str, Any]] = {}

        current_month: str | None = None
        streak_month: str | None = None
        streak_start_day: datetime.date | None = None
        streak_len = 0

        for row in dominant.itertuples(index=False):
            day = row.day
            month_key = row.month_key
            share = float(row.share)
            if share < dominance_min_share:
                continue
            if current_month is None:
                current_month = month_key
                continue
            if month_key == current_month:
                streak_month = None
                streak_start_day = None
                streak_len = 0
                continue
            if streak_month != month_key:
                streak_month = month_key
                streak_start_day = day
                streak_len = 1
            else:
                streak_len += 1
            if streak_len >= dominance_min_days and streak_start_day is not None:
                roll_ts = min_ts_map.get((streak_start_day, streak_month))
                if roll_ts is None:
                    continue
                roll_month_dt = datetime.strptime(streak_month + "-01", "%Y-%m-%d")
                close_start = _close_start_from_roll(roll_month_dt, close_start_day)
                close_end = roll_ts
                window_days = (close_end - close_start).total_seconds() / 86400.0

                window = timedelta(hours=window_hours)
                window_rows = rec_df[
                    (rec_df["ts"] >= roll_ts - window)
                    & (rec_df["ts"] <= roll_ts + window)
                ]
                counts = window_rows["process"].value_counts().to_dict()
                total_window = sum(counts.values()) or 1
                indicator_processes = []
                for process, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:10]:
                    share_val = count / total_window
                    indicator_processes.append(
                        {"process": process, "count": int(count), "share": share_val}
                    )
                    stats = indicator_stats.setdefault(
                        process,
                        {"months_present": 0, "total_hits": 0, "share_sum": 0.0},
                    )
                    stats["months_present"] += 1
                    stats["total_hits"] += int(count)
                    stats["share_sum"] += share_val

                close_windows.append(
                    CloseWindow(
                        accounting_month=streak_month,
                        roll_ts=roll_ts,
                        close_start=close_start,
                        close_end=close_end,
                        window_days=window_days,
                        indicator_processes=indicator_processes,
                    )
                )
                current_month = streak_month
                streak_month = None
                streak_start_day = None
                streak_len = 0

        indicator_list = []
        for process, stats in indicator_stats.items():
            if stats["months_present"] < min_indicator_months:
                continue
            indicator_list.append(
                {
                    "process": process,
                    "months_present": stats["months_present"],
                    "total_hits": stats["total_hits"],
                    "avg_share": stats["share_sum"] / max(stats["months_present"], 1),
                }
            )
        indicator_list = sorted(
            indicator_list, key=lambda item: (item["months_present"], item["avg_share"]), reverse=True
        )

        findings: list[dict[str, Any]] = []
        for window in close_windows:
            findings.append(
                {
                    "kind": "close_cycle_roll",
                    "accounting_month": window.accounting_month,
                    "roll_timestamp": window.roll_ts.isoformat(),
                    "close_start": window.close_start.isoformat(),
                    "close_end": window.close_end.isoformat(),
                    "close_window_days": round(window.window_days, 2),
                    "indicator_window_hours": window_hours,
                    "indicator_processes": window.indicator_processes,
                    "measurement_type": "measured",
                    "confidence": 0.8,
                    "evidence": {
                        "dataset_id": ctx.dataset_id or "unknown",
                        "dataset_version_id": ctx.dataset_version_id or "unknown",
                        "row_ids": [],
                        "column_ids": [],
                        "query": None,
                    },
                }
            )

        findings.append(
            {
                "kind": "close_cycle_indicators",
                "indicator_processes": indicator_list,
                "indicator_window_hours": window_hours,
                "measurement_type": "measured",
                "confidence": 0.8,
                "evidence": {
                    "dataset_id": ctx.dataset_id or "unknown",
                    "dataset_version_id": ctx.dataset_version_id or "unknown",
                    "row_ids": [],
                    "column_ids": [],
                    "query": None,
                },
            }
        )

        metrics = {
            "months_detected": len(close_windows),
            "indicator_window_hours": window_hours,
            "indicator_processes": len(indicator_list),
            "close_start_day": close_start_day,
        }

        summary = f"Detected {len(close_windows)} accounting-month roll events using Accounting Month params."

        artifacts_dir = ctx.artifacts_dir("analysis_dynamic_close_detection")
        results_path = artifacts_dir / "results.json"
        write_json(
            results_path,
            {
                "summary": summary,
                "metrics": metrics,
                "findings": findings,
            },
        )

        csv_path = artifacts_dir / "close_windows.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "accounting_month",
                    "close_start",
                    "close_end",
                    "close_window_days",
                ],
            )
            writer.writeheader()
            for window in close_windows:
                writer.writerow(
                    {
                        "accounting_month": window.accounting_month,
                        "close_start": window.close_start.isoformat(),
                        "close_end": window.close_end.isoformat(),
                        "close_window_days": round(window.window_days, 2),
                    }
                )

        artifacts = [
            PluginArtifact(
                path=str(results_path.relative_to(ctx.run_dir)),
                type="json",
                description="Dynamic close window detection results",
            ),
            PluginArtifact(
                path=str(csv_path.relative_to(ctx.run_dir)),
                type="csv",
                description="Close windows per accounting month",
            ),
        ]

        return PluginResult("ok", summary, metrics, findings, artifacts, None)
