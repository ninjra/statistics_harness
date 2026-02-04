from __future__ import annotations

from typing import Any

import pandas as pd

from statistic_harness.core.column_inference import infer_timestamp_series
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import infer_close_cycle_window, write_json


INVALID_STRINGS = {"", "nan", "none", "null"}
MAX_SAMPLE_ROWS = 50000
MIN_PROCESS_ALPHA_RATIO = 0.15
MIN_DATETIME_RATIO = 0.15


def _normalize_process(value: Any) -> str:
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


def _parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


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

        end_candidates = _candidate_columns(
            ctx.settings.get("end_column"),
            columns,
            role_by_name,
            {"end", "finish", "complete"},
            ["end", "finish", "complete", "stop"],
            lower_names,
            used,
        )
        end_col = _best_datetime_column(end_candidates, df)
        if end_col:
            used.add(end_col)

        if not process_col:
            return PluginResult(
                "skipped", "No process/activity column detected", {}, [], [], None
            )
        if not start_col or (not queue_col and not eligible_col):
            return PluginResult(
                "skipped",
                "Queue/start timestamps required for delay decomposition",
                {},
                [],
                [],
                None,
            )

        dep_cols = _parse_list(ctx.settings.get("dependency_columns"))
        if not dep_cols:
            dep_tokens = ("dep", "dependency", "parent", "master", "preced")
            dep_cols = [
                col for col in columns if any(tok in lower_names[col] for tok in dep_tokens)
            ]

        selected_cols: list[str] = []
        for col in [process_col, queue_col, eligible_col, start_col, end_col, *dep_cols]:
            if col and col in columns and col not in selected_cols:
                selected_cols.append(col)
        work = df.loc[:, selected_cols].copy()

        work["__process"] = work[process_col].map(_normalize_process)
        work["__process_norm"] = work["__process"].str.lower()
        work = work.loc[~work["__process_norm"].isin(INVALID_STRINGS)].copy()
        if work.empty:
            return PluginResult("skipped", "No valid process values", {}, [], [], None)

        queue_ts = pd.to_datetime(work[queue_col], errors="coerce", utc=False) if queue_col else None
        eligible_ts = pd.to_datetime(work[eligible_col], errors="coerce", utc=False) if eligible_col else None
        start_ts = pd.to_datetime(work[start_col], errors="coerce", utc=False)
        if eligible_ts is None:
            eligible_ts = queue_ts
            eligible_basis = "queue"
        else:
            eligible_basis = "eligible"

        if queue_ts is None and eligible_ts is None:
            return PluginResult("skipped", "No queue/eligible timestamps", {}, [], [], None)

        work["__start_ts"] = start_ts
        work["__queue_ts"] = queue_ts if queue_ts is not None else eligible_ts
        work["__eligible_ts"] = eligible_ts
        work = work.loc[work["__start_ts"].notna() & work["__eligible_ts"].notna()].copy()
        if work.empty:
            return PluginResult("skipped", "No valid timestamps found", {}, [], [], None)

        wait_pre = (work["__eligible_ts"] - work["__queue_ts"]).dt.total_seconds()
        wait_pre = wait_pre.clip(lower=0).fillna(0)
        eligible_wait = (work["__start_ts"] - work["__eligible_ts"]).dt.total_seconds()
        eligible_wait = eligible_wait.clip(lower=0).fillna(0)

        work["__wait_pre_sec"] = wait_pre
        work["__eligible_wait_sec"] = eligible_wait

        threshold = float(ctx.settings.get("wait_threshold_seconds", 60))
        work["__eligible_wait_gt_sec"] = eligible_wait.where(eligible_wait > threshold, 0.0)
        wait_to_start = (work["__start_ts"] - work["__queue_ts"]).dt.total_seconds()
        wait_to_start = wait_to_start.clip(lower=0).fillna(0)
        work["__wait_to_start_sec"] = wait_to_start
        work["__wait_to_start_gt_sec"] = wait_to_start.where(wait_to_start > threshold, 0.0)

        close_mode = str(ctx.settings.get("close_cycle_mode", "infer")).lower()
        window_days = int(ctx.settings.get("close_cycle_window_days", 17))
        inferred_start, inferred_end = infer_close_cycle_window(
            work["__start_ts"], window_days
        )
        if close_mode == "fixed":
            close_start = int(ctx.settings.get("close_cycle_start_day", inferred_start))
            close_end = int(ctx.settings.get("close_cycle_end_day", inferred_end))
            close_source = "fixed"
        else:
            close_start = inferred_start
            close_end = inferred_end
            close_source = "inferred"
        work["__day"] = work["__start_ts"].dt.day
        if close_start <= close_end:
            work["__close"] = (work["__day"] >= close_start) & (work["__day"] <= close_end)
        else:
            work["__close"] = (work["__day"] >= close_start) | (work["__day"] <= close_end)

        sequence_mask = pd.Series(False, index=work.index)
        for col in dep_cols:
            if col not in work.columns:
                continue
            series = work[col]
            mask = series.notna()
            if mask.any():
                text = series.astype(str).str.strip().str.lower()
                mask = mask & (~text.isin(INVALID_STRINGS))
            sequence_mask |= mask
        work["__sequence"] = sequence_mask
        work["__standalone"] = ~sequence_mask

        exclude_list = _parse_list(ctx.settings.get("exclude_processes"))
        if not exclude_list:
            exclude_list = ["qlongjob"]
        exclude_processes = {p.lower() for p in exclude_list}
        work["__excluded"] = work["__process_norm"].isin(exclude_processes)

        standalone = work.loc[work["__standalone"] & ~work["__excluded"]].copy()
        if standalone.empty:
            return PluginResult(
                "ok",
                "No standalone rows after filtering",
                {
                    "standalone_runs": 0,
                    "sequence_runs": int(sequence_mask.sum()),
                    "eligible_basis": eligible_basis,
                },
                [],
                [],
                None,
            )

        busy_periods: list[dict[str, Any]] = []
        busy_frame = standalone.loc[standalone["__close"]].copy()
        grouped = pd.DataFrame()
        if not busy_frame.empty:
            bucket_ts = busy_frame["__queue_ts"].dt.floor("h")
            grouped = (
                busy_frame.groupby(bucket_ts)
                .agg(
                    rows_total=("__queue_ts", "size"),
                    rows_over_threshold=(
                        "__wait_to_start_gt_sec",
                        lambda series: int((series > 0).sum()),
                    ),
                    wait_to_start_hours_total=("__wait_to_start_gt_sec", "sum"),
                )
                .reset_index()
                .rename(columns={"__queue_ts": "bucket_start"})
            )
        if not grouped.empty:
            grouped["wait_to_start_hours_total"] = grouped["wait_to_start_hours_total"] / 3600.0
            grouped["bucket_end"] = grouped["bucket_start"] + pd.Timedelta(hours=1)
            grouped["weekday"] = grouped["bucket_start"].dt.day_name()
            grouped["weekend"] = grouped["bucket_start"].dt.dayofweek >= 5
            grouped["hour"] = grouped["bucket_start"].dt.hour
            grouped["after_hours"] = (grouped["hour"] < 8) | (grouped["hour"] >= 18)
            grouped = grouped.sort_values(
                ["wait_to_start_hours_total", "bucket_start"],
                ascending=[False, True],
            )
            for _, row in grouped.iterrows():
                busy_periods.append(
                    {
                        "period_start": row["bucket_start"].isoformat(),
                        "period_end": row["bucket_end"].isoformat(),
                        "wait_to_start_hours_total": float(row["wait_to_start_hours_total"]),
                        "rows_total": int(row["rows_total"]),
                        "rows_over_threshold": int(row["rows_over_threshold"]),
                        "weekday": row["weekday"],
                        "weekend": bool(row["weekend"]),
                        "after_hours": bool(row["after_hours"]),
                    }
                )

        agg = (
            standalone.groupby(["__process_norm", "__close"], dropna=False)
            .agg(
                runs=("__process_norm", "size"),
                eligible_wait_sec=("__eligible_wait_sec", "sum"),
                eligible_wait_gt_sec=("__eligible_wait_gt_sec", "sum"),
                wait_pre_sec=("__wait_pre_sec", "sum"),
            )
            .reset_index()
        )

        stats: dict[str, dict[str, float]] = {}
        for _, row in agg.iterrows():
            proc = row["__process_norm"]
            close_flag = bool(row["__close"])
            entry = stats.setdefault(proc, {})
            suffix = "close" if close_flag else "open"
            entry[f"runs_{suffix}"] = float(row["runs"])
            entry[f"eligible_wait_sec_{suffix}"] = float(row["eligible_wait_sec"])
            entry[f"eligible_wait_gt_sec_{suffix}"] = float(row["eligible_wait_gt_sec"])
            entry[f"wait_pre_sec_{suffix}"] = float(row["wait_pre_sec"])

        process_labels = (
            standalone.groupby("__process_norm")["__process"]
            .agg(lambda series: series.value_counts().index[0])
            .to_dict()
        )

        summaries = []
        for proc, values in stats.items():
            runs_close = values.get("runs_close", 0.0)
            runs_open = values.get("runs_open", 0.0)
            eligible_wait_sec_close = values.get("eligible_wait_sec_close", 0.0)
            eligible_wait_sec_open = values.get("eligible_wait_sec_open", 0.0)
            eligible_wait_gt_sec_close = values.get("eligible_wait_gt_sec_close", 0.0)
            eligible_wait_gt_sec_open = values.get("eligible_wait_gt_sec_open", 0.0)
            wait_pre_sec_close = values.get("wait_pre_sec_close", 0.0)
            wait_pre_sec_open = values.get("wait_pre_sec_open", 0.0)
            summaries.append(
                {
                    "process": process_labels.get(proc, proc),
                    "process_norm": proc,
                    "runs_total": runs_close + runs_open,
                    "runs_close": runs_close,
                    "runs_open": runs_open,
                    "eligible_wait_hours_total": (eligible_wait_sec_close + eligible_wait_sec_open) / 3600.0,
                    "eligible_wait_hours_close": eligible_wait_sec_close / 3600.0,
                    "eligible_wait_hours_open": eligible_wait_sec_open / 3600.0,
                    "eligible_wait_gt_hours_total": (eligible_wait_gt_sec_close + eligible_wait_gt_sec_open) / 3600.0,
                    "eligible_wait_gt_hours_close": eligible_wait_gt_sec_close / 3600.0,
                    "eligible_wait_gt_hours_open": eligible_wait_gt_sec_open / 3600.0,
                    "wait_pre_hours_total": (wait_pre_sec_close + wait_pre_sec_open) / 3600.0,
                }
            )

        summaries = sorted(
            summaries,
            key=lambda item: (-(item["eligible_wait_hours_total"]), item["process_norm"]),
        )

        min_hours = float(ctx.settings.get("min_total_hours", 1))
        max_findings = int(ctx.settings.get("max_process_findings", 5))
        target_process = str(ctx.settings.get("target_process") or "qemail").strip().lower()
        findings = []
        columns_used = [
            col
            for col in [process_col, queue_col, eligible_col, start_col, end_col]
            if col
        ]

        def build_finding(entry: dict[str, float]) -> dict[str, Any]:
            return {
                "kind": "eligible_wait_process_stats",
                "process": entry["process"],
                "process_norm": entry["process_norm"],
                "runs_total": int(entry["runs_total"]),
                "runs_close": int(entry["runs_close"]),
                "runs_open": int(entry["runs_open"]),
                "eligible_wait_hours_total": float(entry["eligible_wait_hours_total"]),
                "eligible_wait_hours_close": float(entry["eligible_wait_hours_close"]),
                "eligible_wait_hours_open": float(entry["eligible_wait_hours_open"]),
                "eligible_wait_gt_hours_total": float(entry["eligible_wait_gt_hours_total"]),
                "eligible_wait_gt_hours_close": float(entry["eligible_wait_gt_hours_close"]),
                "eligible_wait_gt_hours_open": float(entry["eligible_wait_gt_hours_open"]),
                "wait_pre_hours_total": float(entry["wait_pre_hours_total"]),
                "close_cycle_start_day": close_start,
                "close_cycle_end_day": close_end,
                "eligible_basis": eligible_basis,
                "columns": columns_used,
            }

        target_entry = None
        for entry in summaries:
            if target_process and entry["process_norm"] == target_process:
                target_entry = entry
                break
        if target_entry:
            finding = build_finding(target_entry)
            finding["measurement_type"] = "measured"
            target_rows = work.loc[
                (work["__process_norm"] == target_process) & work["__close"]
            ].index.tolist()
            finding["row_ids"] = [int(i) for i in target_rows[: int(ctx.settings.get("max_examples", 25))]]
            findings.append(finding)

        for entry in summaries:
            if entry["eligible_wait_hours_total"] < min_hours:
                continue
            if target_process and entry["process_norm"] == target_process:
                continue
            finding = build_finding(entry)
            finding["measurement_type"] = "measured"
            findings.append(finding)
            if len(findings) >= max_findings + (1 if target_entry else 0):
                break

        total_runs = int(standalone.shape[0])
        sequence_runs = int(work["__sequence"].sum())

        total_eligible_wait_hours = float(standalone["__eligible_wait_sec"].sum() / 3600.0)
        total_eligible_wait_gt_hours = float(standalone["__eligible_wait_gt_sec"].sum() / 3600.0)
        total_close_hours = float(
            standalone.loc[standalone["__close"], "__eligible_wait_sec"].sum() / 3600.0
        )
        total_open_hours = float(
            standalone.loc[~standalone["__close"], "__eligible_wait_sec"].sum() / 3600.0
        )
        total_close_gt_hours = float(
            standalone.loc[standalone["__close"], "__eligible_wait_gt_sec"].sum()
            / 3600.0
        )
        total_open_gt_hours = float(
            standalone.loc[~standalone["__close"], "__eligible_wait_gt_sec"].sum()
            / 3600.0
        )

        impact_finding = None
        if target_entry:
            remaining_hours = total_eligible_wait_hours - float(
                target_entry["eligible_wait_hours_total"]
            )
            remaining_gt_hours = total_eligible_wait_gt_hours - float(
                target_entry["eligible_wait_gt_hours_total"]
            )
            remaining_close_hours = total_close_hours - float(
                target_entry["eligible_wait_hours_close"]
            )
            remaining_open_hours = total_open_hours - float(
                target_entry["eligible_wait_hours_open"]
            )
            remaining_close_gt_hours = total_close_gt_hours - float(
                target_entry["eligible_wait_gt_hours_close"]
            )
            remaining_open_gt_hours = total_open_gt_hours - float(
                target_entry["eligible_wait_gt_hours_open"]
            )
            impact_finding = {
                "kind": "eligible_wait_impact",
                "process": target_entry["process"],
                "eligible_wait_hours_total": total_eligible_wait_hours,
                "eligible_wait_hours_without_target": remaining_hours,
                "eligible_wait_hours_close_total": total_close_hours,
                "eligible_wait_hours_close_without_target": remaining_close_hours,
                "eligible_wait_hours_open_total": total_open_hours,
                "eligible_wait_hours_open_without_target": remaining_open_hours,
                "eligible_wait_gt_hours_total": total_eligible_wait_gt_hours,
                "eligible_wait_gt_hours_without_target": remaining_gt_hours,
                "eligible_wait_gt_hours_close_total": total_close_gt_hours,
                "eligible_wait_gt_hours_close_without_target": remaining_close_gt_hours,
                "eligible_wait_gt_hours_open_total": total_open_gt_hours,
                "eligible_wait_gt_hours_open_without_target": remaining_open_gt_hours,
                "measurement_type": "measured",
                "columns": columns_used,
            }
            findings.append(impact_finding)

            scale_factor = float(ctx.settings.get("capacity_scale_factor", 0.6667))
            if remaining_gt_hours > 0 and scale_factor > 0:
                modeled = remaining_gt_hours * scale_factor
                modeled_close = remaining_close_gt_hours * scale_factor
                modeled_open = remaining_open_gt_hours * scale_factor
                assumptions = [
                    "capacity-proportional scaling on >threshold eligible-wait",
                ]
                scope = {
                    "metric": "eligible_wait_gt_hours",
                    "close_cycle_start_day": close_start,
                    "close_cycle_end_day": close_end,
                    "close_cycle_mode": close_mode,
                    "close_cycle_window_days": window_days,
                    "close_cycle_source": close_source,
                    "inferred_close_cycle_start_day": inferred_start,
                    "inferred_close_cycle_end_day": inferred_end,
                    "eligible_basis": eligible_basis,
                }
                findings.append(
                    {
                        "kind": "capacity_scale_model",
                        "process": target_entry["process"],
                        "eligible_wait_gt_hours_without_target": remaining_gt_hours,
                        "eligible_wait_gt_hours_modeled": modeled,
                        "eligible_wait_gt_hours_close_without_target": remaining_close_gt_hours,
                        "eligible_wait_gt_hours_close_modeled": modeled_close,
                        "eligible_wait_gt_hours_open_without_target": remaining_open_gt_hours,
                        "eligible_wait_gt_hours_open_modeled": modeled_open,
                        "scale_factor": scale_factor,
                        "host_count_baseline": None,
                        "host_count_modeled": None,
                        "measurement_type": "modeled",
                        "assumptions": assumptions,
                        "scope": scope,
                        "columns": columns_used,
                    }
                )

        artifacts_dir = ctx.artifacts_dir("analysis_queue_delay_decomposition")
        out_path = artifacts_dir / "results.json"
        write_json(
            out_path,
            {
                "summary": {
                    "process_column": process_col,
                    "queue_column": queue_col,
                    "eligible_column": eligible_col,
                    "start_column": start_col,
                    "end_column": end_col,
                    "dependency_columns": dep_cols,
                    "eligible_basis": eligible_basis,
                    "close_cycle_start_day": close_start,
                    "close_cycle_end_day": close_end,
                    "close_cycle_mode": close_mode,
                    "close_cycle_window_days": window_days,
                    "close_cycle_source": close_source,
                    "inferred_close_cycle_start_day": inferred_start,
                    "inferred_close_cycle_end_day": inferred_end,
                    "wait_threshold_seconds": threshold,
                    "busy_period_bucket": "hour",
                    "busy_period_basis": "queue_to_start",
                    "busy_period_scope": "close_cycle",
                },
                "process_stats": summaries,
                "totals": {
                    "standalone_runs": total_runs,
                    "sequence_runs": sequence_runs,
                    "eligible_wait_hours_total": total_eligible_wait_hours,
                    "eligible_wait_gt_hours_total": total_eligible_wait_gt_hours,
                },
                "busy_periods": busy_periods,
            },
        )
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Queue delay decomposition results",
            )
        ]

        return PluginResult(
            "ok",
            "Computed queue delay decomposition",
            {
                "standalone_runs": total_runs,
                "sequence_runs": sequence_runs,
                "eligible_wait_hours_total": total_eligible_wait_hours,
                "eligible_wait_gt_hours_total": total_eligible_wait_gt_hours,
                "eligible_basis": eligible_basis,
                "process_column": process_col,
                "close_cycle_start_day": close_start,
                "close_cycle_end_day": close_end,
                "close_cycle_mode": close_mode,
                "close_cycle_window_days": window_days,
                "close_cycle_source": close_source,
                "inferred_close_cycle_start_day": inferred_start,
                "inferred_close_cycle_end_day": inferred_end,
            },
            findings,
            artifacts,
            None,
        )
