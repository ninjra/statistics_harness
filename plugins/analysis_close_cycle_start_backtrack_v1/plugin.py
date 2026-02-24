from __future__ import annotations

import calendar
import csv
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from statistic_harness.core.column_inference import infer_timestamp_series
from statistic_harness.core.accounting_windows import parse_accounting_month_from_params
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json

PARAM_PATTERN = re.compile(r"\s*([^;=]+?)\s*\([^\)]*\)\s*=\s*([^;]+)")


@dataclass(frozen=True)
class MonthRoll:
    accounting_month: str
    roll_ts: datetime
    close_start_default: datetime
    close_end_default: datetime


@dataclass(frozen=True)
class BacktrackedWindow:
    accounting_month: str
    roll_ts: datetime
    close_start_default: datetime
    close_end_default: datetime
    close_start_dynamic: datetime
    close_end_dynamic: datetime
    source: str
    confidence: float
    fallback_reason: str | None
    signature_processes: list[str]
    signature_coverage: float


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


def _detect_month_rolls(
    rec_df: pd.DataFrame,
    *,
    close_start_day: int,
    close_end_day: int,
    dominance_min_share: float,
    persistence_days: int,
) -> list[MonthRoll]:
    rec_df = rec_df.copy()
    rec_df["month_key"] = rec_df["acct_month"].dt.strftime("%Y-%m")
    rec_df["day"] = rec_df["ts"].dt.date

    day_counts = rec_df.groupby(["day", "month_key"]).size().reset_index(name="count")
    totals = day_counts.groupby("day")["count"].sum().reset_index(name="total")
    merged = day_counts.merge(totals, on="day", how="left")
    merged["share"] = merged["count"] / merged["total"].clip(lower=1)

    min_ts_map = rec_df.groupby(["day", "month_key"])["ts"].min().to_dict()

    month_rolls: list[MonthRoll] = []
    for month_key in sorted(rec_df["month_key"].unique()):
        month_days = merged[merged["month_key"] == month_key].sort_values("day")
        if month_days.empty:
            continue
        candidates = month_days[month_days["share"] >= dominance_min_share]
        roll_day: date | None = None
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
        month_rolls.append(
            MonthRoll(
                accounting_month=month_key,
                roll_ts=roll_ts,
                close_start_default=_close_start_from_roll(roll_month_dt, close_start_day),
                close_end_default=_safe_date(roll_month_dt.year, roll_month_dt.month, close_end_day),
            )
        )
    return month_rolls


def _signature_processes(
    rec_df: pd.DataFrame,
    month_rolls: list[MonthRoll],
    *,
    lookback_days: int,
    min_signature_months: int,
    min_signature_share: float,
) -> list[str]:
    months_present: dict[str, int] = {}
    total_share: dict[str, float] = {}
    for roll in month_rolls:
        lo = roll.roll_ts - timedelta(days=lookback_days)
        span = rec_df[(rec_df["ts"] >= lo) & (rec_df["ts"] <= roll.roll_ts)]
        if span.empty:
            continue
        proc_counts = span["process"].value_counts()
        total = int(proc_counts.sum()) or 1
        for process, count in proc_counts.items():
            share = float(count) / float(total)
            if share < min_signature_share:
                continue
            months_present[process] = months_present.get(process, 0) + 1
            total_share[process] = total_share.get(process, 0.0) + share

    signatures = [
        (proc, months_present.get(proc, 0), total_share.get(proc, 0.0))
        for proc in months_present
        if months_present.get(proc, 0) >= min_signature_months
    ]
    signatures.sort(key=lambda item: (item[1], item[2], item[0]), reverse=True)
    return [proc for proc, _, _ in signatures]


def _infer_start_for_month(
    rec_df: pd.DataFrame,
    roll: MonthRoll,
    signatures: list[str],
    *,
    lookback_days: int,
    min_streak_days: int,
    max_gap_days: int,
) -> tuple[datetime, str, float, float]:
    if not signatures:
        return roll.close_start_default, "fallback_no_signatures", 0.35, 0.0

    lo = roll.roll_ts - timedelta(days=lookback_days)
    span = rec_df[(rec_df["ts"] >= lo) & (rec_df["ts"] <= roll.roll_ts)]
    if span.empty:
        return roll.close_start_default, "fallback_no_rows_in_lookback", 0.35, 0.0

    signature_rows = span[span["process"].isin(signatures)]
    if signature_rows.empty:
        return roll.close_start_default, "fallback_no_signature_rows", 0.35, 0.0

    per_day = (
        signature_rows.groupby(signature_rows["ts"].dt.date)
        .size()
        .reset_index(name="hits")
        .sort_values("ts")
    )
    active_days = [d for d in per_day["ts"].tolist() if d <= roll.roll_ts.date()]
    if not active_days:
        return roll.close_start_default, "fallback_no_active_days", 0.35, 0.0

    # Build a backward streak ending at the latest active day before roll.
    current = active_days[-1]
    streak: list[date] = [current]
    for day in reversed(active_days[:-1]):
        gap = (current - day).days
        if gap <= (max_gap_days + 1):
            streak.append(day)
            current = day
            continue
        break

    if len(streak) < min_streak_days:
        return roll.close_start_default, "fallback_short_streak", 0.45, 0.0

    start_day = min(streak)
    close_start_dynamic = datetime.combine(start_day, datetime.min.time())

    signature_coverage = float(len(streak)) / float(max(1, lookback_days))
    confidence = 0.55 + min(0.35, signature_coverage)
    return close_start_dynamic, "backtracked_signature", confidence, signature_coverage


class Plugin:
    def run(self, ctx) -> PluginResult:
        df = ctx.dataset_loader()
        if df.empty:
            return PluginResult("skipped", "Empty dataset", {}, [], [], None)

        role_by_name: dict[str, str] = {}
        if ctx.dataset_version_id:
            dataset_template = ctx.storage.fetch_dataset_template(ctx.dataset_version_id)
            if dataset_template and dataset_template.get("status") == "ready":
                fields = ctx.storage.fetch_template_fields(int(dataset_template["template_id"]))
                role_by_name = {field["name"]: (field.get("role") or "") for field in fields}
            else:
                columns_meta = ctx.storage.fetch_dataset_columns(ctx.dataset_version_id)
                role_by_name = {
                    col["original_name"]: (col.get("role") or "") for col in columns_meta
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
            return PluginResult("ok", "Close-start backtracking not applicable", {}, [], [], None)

        process_series = df[process_col].astype(str).str.strip().str.lower()
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
                    "process": process_series.iloc[idx],
                    "acct_month": acct_month,
                    "ts": stamp.to_pydatetime(),
                }
            )

        if not records:
            return PluginResult(
                "ok",
                "No accounting month markers found for close-start backtracking.",
                {"months_detected": 0},
                [],
                [],
                None,
            )

        close_start_day = int(ctx.settings.get("close_start_day") or 20)
        close_end_day = int(ctx.settings.get("close_end_day") or 5)
        lookback_days = int(ctx.settings.get("lookback_days") or 21)
        min_signature_months = int(ctx.settings.get("min_signature_months") or 3)
        min_signature_share = float(ctx.settings.get("min_signature_share") or 0.15)
        min_streak_days = int(ctx.settings.get("min_streak_days") or 3)
        max_gap_days = int(ctx.settings.get("max_gap_days") or 1)
        dominance_min_share = float(ctx.settings.get("dominance_min_share") or 0.5)
        persistence_days = int(ctx.settings.get("persistence_days") or 2)
        indicator_window_hours = float(ctx.settings.get("indicator_window_hours") or 24)

        rec_df = pd.DataFrame(records)

        month_rolls = _detect_month_rolls(
            rec_df,
            close_start_day=close_start_day,
            close_end_day=close_end_day,
            dominance_min_share=dominance_min_share,
            persistence_days=persistence_days,
        )

        if not month_rolls:
            return PluginResult(
                "ok",
                "No month-roll events detected for close-start backtracking.",
                {"months_detected": 0},
                [],
                [],
                None,
            )

        signatures = _signature_processes(
            rec_df,
            month_rolls,
            lookback_days=lookback_days,
            min_signature_months=min_signature_months,
            min_signature_share=min_signature_share,
        )

        windows: list[BacktrackedWindow] = []
        findings: list[dict[str, Any]] = []
        fallback_months = 0
        for roll in month_rolls:
            start_dynamic, source, confidence, signature_coverage = _infer_start_for_month(
                rec_df,
                roll,
                signatures,
                lookback_days=lookback_days,
                min_streak_days=min_streak_days,
                max_gap_days=max_gap_days,
            )
            fallback_reason = source if source.startswith("fallback_") else None
            if fallback_reason:
                fallback_months += 1
            end_dynamic = roll.roll_ts
            dynamic_days = (end_dynamic - start_dynamic).total_seconds() / 86400.0
            end_delta_days = (end_dynamic - roll.close_end_default).total_seconds() / 86400.0

            window = BacktrackedWindow(
                accounting_month=roll.accounting_month,
                roll_ts=roll.roll_ts,
                close_start_default=roll.close_start_default,
                close_end_default=roll.close_end_default,
                close_start_dynamic=start_dynamic,
                close_end_dynamic=end_dynamic,
                source=source,
                confidence=float(round(confidence, 4)),
                fallback_reason=fallback_reason,
                signature_processes=signatures,
                signature_coverage=float(round(signature_coverage, 4)),
            )
            windows.append(window)

            findings.append(
                {
                    "kind": "close_cycle_start_backtracked",
                    "accounting_month": window.accounting_month,
                    "roll_timestamp": window.roll_ts.isoformat(),
                    "close_start_default": window.close_start_default.isoformat(),
                    "close_end_default": window.close_end_default.isoformat(),
                    "close_start_dynamic": window.close_start_dynamic.isoformat(),
                    "close_end_dynamic": window.close_end_dynamic.isoformat(),
                    "close_window_days_dynamic": round(dynamic_days, 2),
                    "close_end_delta_days": round(end_delta_days, 2),
                    "source": window.source,
                    "confidence": window.confidence,
                    "fallback_reason": window.fallback_reason,
                    "signature_processes": window.signature_processes,
                    "signature_coverage": window.signature_coverage,
                    "indicator_window_hours": indicator_window_hours,
                    "measurement_type": "measured",
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
            "months_detected": len(windows),
            "months_backtracked": len(windows) - fallback_months,
            "months_fallback": fallback_months,
            "close_start_day": close_start_day,
            "close_end_day": close_end_day,
            "lookback_days": lookback_days,
            "min_signature_months": min_signature_months,
            "min_signature_share": min_signature_share,
            "min_streak_days": min_streak_days,
            "max_gap_days": max_gap_days,
            "signature_processes": len(signatures),
        }

        summary = (
            f"Backtracked close starts for {len(windows)} accounting months "
            f"({len(windows) - fallback_months} inferred, {fallback_months} fallback)."
        )

        artifacts_dir = ctx.artifacts_dir("analysis_close_cycle_start_backtrack_v1")
        results_path = artifacts_dir / "results.json"
        write_json(
            results_path,
            {
                "summary": summary,
                "metrics": metrics,
                "findings": findings,
                "signature_processes": signatures,
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
                    "close_window_days_dynamic",
                    "close_end_delta_days",
                    "source",
                    "confidence",
                    "fallback_reason",
                    "signature_processes",
                    "signature_coverage",
                ],
            )
            writer.writeheader()
            for window in windows:
                close_window_days_dynamic = (
                    window.close_end_dynamic - window.close_start_dynamic
                ).total_seconds() / 86400.0
                close_end_delta_days = (
                    window.close_end_dynamic - window.close_end_default
                ).total_seconds() / 86400.0
                writer.writerow(
                    {
                        "accounting_month": window.accounting_month,
                        "roll_timestamp": window.roll_ts.isoformat(),
                        "close_start_default": window.close_start_default.isoformat(),
                        "close_end_default": window.close_end_default.isoformat(),
                        "close_start_dynamic": window.close_start_dynamic.isoformat(),
                        "close_end_dynamic": window.close_end_dynamic.isoformat(),
                        "close_window_days_dynamic": round(close_window_days_dynamic, 2),
                        "close_end_delta_days": round(close_end_delta_days, 2),
                        "source": window.source,
                        "confidence": window.confidence,
                        "fallback_reason": window.fallback_reason or "",
                        "signature_processes": ",".join(window.signature_processes),
                        "signature_coverage": window.signature_coverage,
                    }
                )

        artifacts = [
            PluginArtifact(
                path=str(results_path.relative_to(ctx.run_dir)),
                type="json",
                description="Backtracked close-start inference results",
            ),
            PluginArtifact(
                path=str(csv_path.relative_to(ctx.run_dir)),
                type="csv",
                description="Backtracked close windows by accounting month",
            ),
        ]

        return PluginResult("ok", summary, metrics, findings, artifacts, None)
