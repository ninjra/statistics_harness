from __future__ import annotations

import calendar
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
    close_start_default: datetime
    close_end_default: datetime
    close_start_dynamic: datetime
    close_end_dynamic: datetime
    default_days: float
    dynamic_days: float
    delta_days: float
    source: str
    confidence: float
    fallback_reason: str | None
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


def _safe_date(year: int, month: int, day: int) -> datetime:
    last_day = calendar.monthrange(year, month)[1]
    safe_day = min(max(day, 1), last_day)
    return datetime(year, month, safe_day)


def _close_start_from_roll(roll_month: datetime, day: int) -> datetime:
    prev_month = _previous_month_start(roll_month)
    return _safe_date(prev_month.year, prev_month.month, day)


def _load_backtracked_windows(run_dir) -> dict[str, dict[str, Any]]:
    path = (
        run_dir
        / "artifacts"
        / "analysis_close_cycle_start_backtrack_v1"
        / "close_windows.csv"
    )
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                month = str(row.get("accounting_month") or "").strip()
                if not month:
                    continue
                out[month] = row
    except Exception:
        return {}
    return out


def _parse_iso_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if dt is None or pd.isna(dt):
        return None
    return dt.to_pydatetime()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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
                "Close cycle resolver not applicable",
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
                "No accounting month markers found for close cycle resolution.",
                {"months_detected": 0},
                [],
                [],
                None,
            )

        window_hours = float(ctx.settings.get("indicator_window_hours") or 24)
        close_start_day = int(ctx.settings.get("close_start_day") or 20)
        close_end_day = int(ctx.settings.get("close_end_day") or 5)
        min_indicator_months = int(ctx.settings.get("indicator_min_months") or 3)
        dominance_min_share = float(ctx.settings.get("dominance_min_share") or 0.5)
        persistence_days = int(ctx.settings.get("persistence_days") or 2)
        backtracked_by_month = _load_backtracked_windows(ctx.run_dir)

        rec_df = pd.DataFrame(records)
        rec_df["month_key"] = rec_df["acct_month"].dt.strftime("%Y-%m")
        rec_df["day"] = rec_df["ts"].dt.date

        day_counts = (
            rec_df.groupby(["day", "month_key"]).size().reset_index(name="count")
        )
        totals = day_counts.groupby("day")["count"].sum().reset_index(name="total")
        merged = day_counts.merge(totals, on="day", how="left")
        merged["share"] = merged["count"] / merged["total"].clip(lower=1)

        min_ts_map = (
            rec_df.groupby(["day", "month_key"])["ts"]
            .min()
            .to_dict()
        )

        indicator_stats: dict[str, dict[str, Any]] = {}
        close_windows: list[CloseWindow] = []

        month_keys = sorted(rec_df["month_key"].unique())
        for month_key in month_keys:
            month_days = merged[merged["month_key"] == month_key].sort_values("day")
            if month_days.empty:
                continue
            candidates = month_days[month_days["share"] >= dominance_min_share]
            roll_day = None
            if not candidates.empty:
                streak_start = None
                streak_len = 0
                prev_day = None
                for day in candidates["day"]:
                    if prev_day is None or (day - prev_day).days == 1:
                        if streak_start is None:
                            streak_start = day
                            streak_len = 1
                        else:
                            streak_len += 1
                    else:
                        streak_start = day
                        streak_len = 1
                    if streak_len >= persistence_days:
                        roll_day = streak_start
                        break
                    prev_day = day
            if roll_day is None:
                roll_day = month_days["day"].min()

            roll_ts = min_ts_map.get((roll_day, month_key))
            if roll_ts is None:
                continue

            roll_month_dt = datetime.strptime(month_key + "-01", "%Y-%m-%d")
            close_start_default = _close_start_from_roll(roll_month_dt, close_start_day)
            close_end_default = _safe_date(
                roll_month_dt.year, roll_month_dt.month, close_end_day
            )
            close_start_dynamic = close_start_default
            close_end_dynamic = roll_ts
            source = "resolver_roll_default"
            confidence = 0.85
            fallback_reason = None

            backtracked_row = backtracked_by_month.get(month_key)
            if isinstance(backtracked_row, dict):
                start_candidate = _parse_iso_dt(backtracked_row.get("close_start_dynamic"))
                end_candidate = _parse_iso_dt(backtracked_row.get("close_end_dynamic"))
                if start_candidate is not None:
                    close_start_dynamic = start_candidate
                    source = str(backtracked_row.get("source") or "backtracked_signature")
                    confidence = _safe_float(backtracked_row.get("confidence"), 0.85)
                    fallback_text = str(backtracked_row.get("fallback_reason") or "").strip()
                    fallback_reason = fallback_text or None
                if end_candidate is not None:
                    close_end_dynamic = end_candidate

            default_days = (
                close_end_default - close_start_default
            ).total_seconds() / 86400.0
            dynamic_days = (
                close_end_dynamic - close_start_dynamic
            ).total_seconds() / 86400.0
            delta_days = (
                close_end_dynamic - close_end_default
            ).total_seconds() / 86400.0

            window = timedelta(hours=window_hours)
            window_rows = rec_df[
                (rec_df["ts"] >= roll_ts - window)
                & (rec_df["ts"] <= roll_ts + window)
            ]
            counts = window_rows["process"].value_counts().to_dict()
            total_window = sum(counts.values()) or 1
            indicator_processes = []
            for process, count in sorted(
                counts.items(), key=lambda item: item[1], reverse=True
            )[:10]:
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
                    accounting_month=month_key,
                    roll_ts=roll_ts,
                    close_start_default=close_start_default,
                    close_end_default=close_end_default,
                    close_start_dynamic=close_start_dynamic,
                    close_end_dynamic=close_end_dynamic,
                    default_days=default_days,
                    dynamic_days=dynamic_days,
                    delta_days=delta_days,
                    source=source,
                    confidence=confidence,
                    fallback_reason=fallback_reason,
                    indicator_processes=indicator_processes,
                )
            )

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
            indicator_list,
            key=lambda item: (item["months_present"], item["avg_share"]),
            reverse=True,
        )

        findings: list[dict[str, Any]] = []
        for window in close_windows:
            findings.append(
                {
                    "kind": "close_cycle_window_resolved",
                    "accounting_month": window.accounting_month,
                    "roll_timestamp": window.roll_ts.isoformat(),
                    "close_start_default": window.close_start_default.isoformat(),
                    "close_end_default": window.close_end_default.isoformat(),
                    "close_start_dynamic": window.close_start_dynamic.isoformat(),
                    "close_end_dynamic": window.close_end_dynamic.isoformat(),
                    "close_window_days_default": round(window.default_days, 2),
                    "close_window_days_dynamic": round(window.dynamic_days, 2),
                    "close_end_delta_days": round(window.delta_days, 2),
                    "indicator_window_hours": window_hours,
                    "source": window.source,
                    "fallback_reason": window.fallback_reason,
                    "indicator_processes": window.indicator_processes,
                    "measurement_type": "measured",
                    "confidence": window.confidence,
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
                "confidence": 0.85,
                "evidence": {
                    "dataset_id": ctx.dataset_id or "unknown",
                    "dataset_version_id": ctx.dataset_version_id or "unknown",
                    "row_ids": [],
                    "column_ids": [],
                    "query": None,
                },
            }
        )

        default_days = [w.default_days for w in close_windows]
        dynamic_days = [w.dynamic_days for w in close_windows]
        delta_days = [w.delta_days for w in close_windows]

        metrics = {
            "months_detected": len(close_windows),
            "close_start_day": close_start_day,
            "close_end_day": close_end_day,
            "indicator_window_hours": window_hours,
            "dominance_min_share": dominance_min_share,
            "persistence_days": persistence_days,
            "indicator_processes": len(indicator_list),
            "default_window_days_avg": _safe_avg(default_days),
            "dynamic_window_days_avg": _safe_avg(dynamic_days),
            "close_end_delta_days_avg": _safe_avg(delta_days),
            "backtracked_windows_available": bool(backtracked_by_month),
            "backtracked_windows_used": sum(
                1 for w in close_windows if str(w.source).startswith("backtracked")
            ),
            "resolver_fallback_windows": sum(1 for w in close_windows if w.fallback_reason),
        }

        summary = (
            f"Resolved {len(close_windows)} close-cycle windows using Accounting Month roll signals "
            "and default calendar bounds."
        )

        artifacts_dir = ctx.artifacts_dir("analysis_close_cycle_window_resolver")
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
                    "roll_timestamp",
                    "close_start_default",
                    "close_end_default",
                    "close_start_dynamic",
                    "close_end_dynamic",
                    "close_window_days_default",
                    "close_window_days_dynamic",
                    "close_end_delta_days",
                    "source",
                    "confidence",
                    "fallback_reason",
                ],
            )
            writer.writeheader()
            for window in close_windows:
                writer.writerow(
                    {
                        "accounting_month": window.accounting_month,
                        "roll_timestamp": window.roll_ts.isoformat(),
                        "close_start_default": window.close_start_default.isoformat(),
                        "close_end_default": window.close_end_default.isoformat(),
                        "close_start_dynamic": window.close_start_dynamic.isoformat(),
                        "close_end_dynamic": window.close_end_dynamic.isoformat(),
                        "close_window_days_default": round(window.default_days, 2),
                        "close_window_days_dynamic": round(window.dynamic_days, 2),
                        "close_end_delta_days": round(window.delta_days, 2),
                        "source": window.source,
                        "confidence": window.confidence,
                        "fallback_reason": window.fallback_reason or "",
                    }
                )

        artifacts = [
            PluginArtifact(
                path=str(results_path.relative_to(ctx.run_dir)),
                type="json",
                description="Resolved close-cycle windows (default + dynamic)",
            ),
            PluginArtifact(
                path=str(csv_path.relative_to(ctx.run_dir)),
                type="csv",
                description="Close-cycle windows with default/dynamic bounds",
            ),
        ]

        return PluginResult("ok", summary, metrics, findings, artifacts, None)


def _safe_avg(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))
