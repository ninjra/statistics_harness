from __future__ import annotations

from typing import Any

import pandas as pd

from statistic_harness.core.column_inference import choose_timestamp_column
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


INVALID_STRINGS = {"", "nan", "none", "null"}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _pick_column(
    preferred: str | None,
    columns: list[str],
    role_by_name: dict[str, str],
    roles: set[str],
    patterns: list[str],
    lower_names: dict[str, str],
    exclude: set[str],
) -> str | None:
    if preferred and preferred in columns:
        return preferred
    for col in columns:
        if col in exclude:
            continue
        if role_by_name.get(col) in roles:
            return col
    for col in columns:
        if col in exclude:
            continue
        name = lower_names[col]
        if any(pattern in name for pattern in patterns):
            return col
    return None


def _pick_timestamp_column(
    preferred: str | None,
    columns: list[str],
    role_by_name: dict[str, str],
    roles: set[str],
    patterns: list[str],
    lower_names: dict[str, str],
    exclude: set[str],
    df: pd.DataFrame,
) -> str | None:
    candidates: list[str] = []
    if preferred and preferred in columns and preferred not in exclude:
        candidates.append(preferred)
    for col in columns:
        if col in exclude:
            continue
        if role_by_name.get(col) in roles and col not in candidates:
            candidates.append(col)
    for col in columns:
        if col in exclude:
            continue
        name = lower_names[col]
        if any(pattern in name for pattern in patterns) and col not in candidates:
            candidates.append(col)
    if not candidates:
        return None
    return choose_timestamp_column(df, candidates)


def _sweep(events: list[tuple[pd.Timestamp, int]]) -> tuple[int, float]:
    if not events:
        return 0, 0.0
    events = sorted(events, key=lambda item: (item[0], item[1]))
    current = 0
    max_concurrency = 0
    total_weighted = 0.0
    total_duration = 0.0
    prev_time = None
    for ts, delta in events:
        if prev_time is not None and ts > prev_time:
            duration = (ts - prev_time).total_seconds()
            if duration > 0:
                total_weighted += current * duration
                total_duration += duration
        current += delta
        if current > max_concurrency:
            max_concurrency = current
        prev_time = ts
    avg_concurrency = total_weighted / total_duration if total_duration > 0 else 0.0
    return max_concurrency, avg_concurrency


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
        used: set[str] = set()

        start_col = _pick_timestamp_column(
            ctx.settings.get("start_column"),
            columns,
            role_by_name,
            {"start_time", "start"},
            ["start", "begin"],
            lower_names,
            used,
            df,
        )
        if start_col:
            used.add(start_col)

        end_col = _pick_timestamp_column(
            ctx.settings.get("end_column"),
            columns,
            role_by_name,
            {"end_time", "end"},
            ["end", "finish", "complete", "stop"],
            lower_names,
            used,
            df,
        )
        if end_col:
            used.add(end_col)

        host_col = _pick_column(
            ctx.settings.get("host_column"),
            columns,
            role_by_name,
            {"server", "host", "node", "instance"},
            ["server", "host", "node", "instance", "machine"],
            lower_names,
            used,
        )
        if host_col:
            used.add(host_col)

        cap_col = _pick_column(
            ctx.settings.get("capacity_column"),
            columns,
            role_by_name,
            {"capacity", "cap", "limit"},
            ["capacity", "cap", "limit", "slots"],
            lower_names,
            used,
        )

        if not start_col or not end_col:
            return PluginResult(
                "ok",
                "Concurrency reconstruction not applicable",
                {"hosts": 0},
                [
                    {
                        "kind": "concurrency_summary",
                        "measurement_type": "not_applicable",
                        "reason": "Missing start/end columns.",
                        "columns": [col for col in [start_col, end_col, host_col] if col],
                    }
                ],
                [],
                None,
            )

        selected = [col for col in [start_col, end_col, host_col, cap_col] if col]
        work = df.loc[:, selected].copy()
        work["__start_ts"] = pd.to_datetime(work[start_col], errors="coerce", utc=False)
        work["__end_ts"] = pd.to_datetime(work[end_col], errors="coerce", utc=False)
        work = work.loc[work["__start_ts"].notna() & work["__end_ts"].notna()].copy()
        if work.empty:
            return PluginResult(
                "ok",
                "No valid timestamps",
                {"hosts": 0},
                [],
                [],
                None,
            )

        work = work.loc[work["__end_ts"] >= work["__start_ts"]].copy()
        if work.empty:
            return PluginResult(
                "ok",
                "No non-negative durations",
                {"hosts": 0},
                [],
                [],
                None,
            )

        if host_col:
            work["__host"] = work[host_col].map(_normalize_text)
            work["__host_norm"] = work["__host"].str.lower()
            work = work.loc[~work["__host_norm"].isin(INVALID_STRINGS)].copy()
        else:
            work["__host"] = "overall"
            work["__host_norm"] = "overall"

        if work.empty:
            return PluginResult(
                "ok",
                "No valid host values",
                {"hosts": 0},
                [],
                [],
                None,
            )

        capacity_limit = ctx.settings.get("capacity_limit")
        try:
            capacity_limit = float(capacity_limit) if capacity_limit is not None else None
        except (TypeError, ValueError):
            capacity_limit = None

        findings = []
        host_stats = []
        max_hosts = int(ctx.settings.get("max_hosts", 10))
        max_examples = int(ctx.settings.get("max_examples", 25))

        grouped = work.groupby("__host_norm", sort=False)
        for host_norm, frame in grouped:
            host_label = frame["__host"].iloc[0]
            events: list[tuple[pd.Timestamp, int]] = []
            for _, row in frame.iterrows():
                events.append((row["__start_ts"], 1))
                events.append((row["__end_ts"], -1))
            max_conc, avg_conc = _sweep(events)

            host_cap = capacity_limit
            if host_cap is None and cap_col and cap_col in frame.columns:
                values = pd.to_numeric(frame[cap_col], errors="coerce")
                values = values.dropna()
                if not values.empty:
                    host_cap = float(values.median())

            cap_util = None
            cap_exceeded = None
            if host_cap is not None and host_cap > 0:
                cap_util = float(max_conc) / float(host_cap)
                cap_exceeded = bool(max_conc > host_cap)

            row_ids = [int(i) for i in frame.index.tolist()][:max_examples]
            findings.append(
                {
                    "kind": "concurrency_summary",
                    "host": host_label,
                    "peak_concurrency": int(max_conc),
                    "avg_concurrency": float(avg_conc),
                    "cap_limit": host_cap,
                    "cap_utilization": cap_util,
                    "cap_exceeded": cap_exceeded,
                    "measurement_type": "measured",
                    "row_ids": row_ids,
                    "columns": [col for col in [start_col, end_col, host_col, cap_col] if col],
                }
            )
            host_stats.append(
                {
                    "host": host_label,
                    "peak_concurrency": int(max_conc),
                    "avg_concurrency": float(avg_conc),
                    "cap_limit": host_cap,
                }
            )

        findings.sort(key=lambda item: (-item.get("peak_concurrency", 0), str(item.get("host", ""))))
        findings = findings[:max_hosts]

        overall_events: list[tuple[pd.Timestamp, int]] = []
        for _, row in work.iterrows():
            overall_events.append((row["__start_ts"], 1))
            overall_events.append((row["__end_ts"], -1))
        overall_max, overall_avg = _sweep(overall_events)

        metrics = {
            "hosts": int(len(grouped)),
            "peak_concurrency": int(overall_max),
            "avg_concurrency": float(overall_avg),
        }

        artifacts_dir = ctx.artifacts_dir("analysis_concurrency_reconstruction")
        out_path = artifacts_dir / "concurrency_summary.json"
        write_json(
            out_path,
            {
                "summary": {
                    "start_column": start_col,
                    "end_column": end_col,
                    "host_column": host_col,
                    "capacity_column": cap_col,
                },
                "metrics": metrics,
                "hosts": host_stats,
            },
        )
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Concurrency reconstruction summary",
            )
        ]

        return PluginResult(
            "ok",
            "Reconstructed concurrency timeline",
            metrics,
            findings,
            artifacts,
            None,
        )
