from __future__ import annotations

from typing import Any

import pandas as pd

from statistic_harness.core.accounting_windows import (
    infer_accounting_windows_from_timestamps,
    load_accounting_windows_from_run,
    window_ranges,
)
from statistic_harness.core.close_cycle import resolve_close_cycle_masks
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


def _normalize_token(value: Any) -> str:
    token = _normalize_text(value).lower()
    if token in INVALID_STRINGS:
        return ""
    return token


def _pick_column(
    preferred: str | None,
    columns: list[str],
    role_by_name: dict[str, str],
    roles: set[str],
    patterns: tuple[str, ...],
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
    patterns: tuple[str, ...],
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
        if any(pattern in lower_names[col] for pattern in patterns) and col not in candidates:
            candidates.append(col)
    if not candidates:
        return None
    return choose_timestamp_column(df, candidates)


def _mask_from_ranges(ts: pd.Series, ranges: list[tuple[Any, Any]]) -> pd.Series:
    mask = pd.Series(False, index=ts.index)
    if not ranges:
        return mask
    for start, end in ranges:
        if pd.isna(start) or pd.isna(end):
            continue
        mask |= (ts >= start) & (ts <= end)
    return mask


def _window_metrics(wait_seconds: float, delta_seconds: float) -> tuple[float, float]:
    basis_hours = float(wait_seconds) / 3600.0
    delta_hours = float(max(0.0, delta_seconds)) / 3600.0
    pct = 0.0
    if wait_seconds > 0:
        pct = (max(0.0, delta_seconds) / wait_seconds) * 100.0
    return delta_hours, pct


class Plugin:
    def run(self, ctx) -> PluginResult:
        df = ctx.dataset_loader()
        if df.empty:
            return PluginResult(
                "ok",
                "Process/server lever model not applicable: empty dataset",
                {"rows": 0, "not_applicable_reason": "empty_dataset"},
                [
                    {
                        "kind": "process_server_lever",
                        "decision": "not_applicable",
                        "measurement_type": "not_applicable",
                        "reason_code": "EMPTY_DATASET",
                        "reason": "Dataset contains no rows.",
                    }
                ],
                [],
                None,
            )

        role_by_name: dict[str, str] = {}
        if ctx.dataset_version_id:
            template = ctx.storage.fetch_dataset_template(ctx.dataset_version_id)
            if template and template.get("status") == "ready":
                fields = ctx.storage.fetch_template_fields(int(template["template_id"]))
                role_by_name = {
                    field["name"]: (field.get("role") or "") for field in fields
                }
            else:
                columns_meta = ctx.storage.fetch_dataset_columns(ctx.dataset_version_id)
                role_by_name = {
                    col["original_name"]: (col.get("role") or "") for col in columns_meta
                }

        columns = list(df.columns)
        lower_names = {col: str(col).lower() for col in columns}
        used: set[str] = set()

        process_col = _pick_column(
            ctx.settings.get("process_column"),
            columns,
            role_by_name,
            {"process", "process_id", "process_name", "activity", "event", "task"},
            ("process", "activity", "event", "task", "job", "step"),
            lower_names,
            used,
        )
        if process_col:
            used.add(process_col)

        server_col = _pick_column(
            ctx.settings.get("server_column"),
            columns,
            role_by_name,
            {"server", "host", "node", "instance", "machine"},
            ("server", "host", "node", "instance", "machine"),
            lower_names,
            used,
        )
        if server_col:
            used.add(server_col)

        queue_col = _pick_timestamp_column(
            ctx.settings.get("queue_column"),
            columns,
            role_by_name,
            {"queue", "queued", "enqueue"},
            ("queue", "queued", "enqueue"),
            lower_names,
            used,
            df,
        )
        if queue_col:
            used.add(queue_col)

        eligible_col = _pick_timestamp_column(
            ctx.settings.get("eligible_column"),
            columns,
            role_by_name,
            {"eligible", "ready", "available"},
            ("eligible", "ready", "available"),
            lower_names,
            used,
            df,
        )
        if eligible_col:
            used.add(eligible_col)

        start_col = _pick_timestamp_column(
            ctx.settings.get("start_column"),
            columns,
            role_by_name,
            {"start", "start_time", "begin"},
            ("start", "begin", "launched"),
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
            {"end", "finish", "complete"},
            ("end", "finish", "complete", "stop"),
            lower_names,
            used,
            df,
        )
        if end_col:
            used.add(end_col)

        duration_col = _pick_column(
            ctx.settings.get("duration_column"),
            columns,
            role_by_name,
            {"duration", "elapsed", "runtime"},
            ("duration", "elapsed", "runtime", "seconds", "ms"),
            lower_names,
            used,
        )
        if duration_col:
            used.add(duration_col)

        required_missing = [
            key
            for key, col in (
                ("process", process_col),
                ("server", server_col),
                ("start", start_col),
            )
            if not col
        ]
        if required_missing:
            return PluginResult(
                "ok",
                "Process/server lever model not applicable: missing required columns",
                {
                    "rows": int(df.shape[0]),
                    "not_applicable_reason": "missing_required_columns",
                    "missing": required_missing,
                },
                [
                    {
                        "kind": "process_server_lever",
                        "decision": "not_applicable",
                        "measurement_type": "not_applicable",
                        "reason_code": "MISSING_REQUIRED_COLUMNS",
                        "reason": "Missing required process/server/start columns for lever modeling.",
                        "missing": required_missing,
                    }
                ],
                [],
                None,
            )

        selected_cols = [
            col
            for col in (
                process_col,
                server_col,
                queue_col,
                eligible_col,
                start_col,
                end_col,
                duration_col,
            )
            if col
        ]
        work = df.loc[:, selected_cols].copy()
        work["process_norm"] = work[process_col].map(_normalize_token)
        work["server_norm"] = work[server_col].map(_normalize_token)
        work["__start_ts"] = pd.to_datetime(work[start_col], errors="coerce", utc=False)

        eligible_ts: pd.Series | None = None
        wait_signal = ""
        if eligible_col:
            eligible_ts = pd.to_datetime(work[eligible_col], errors="coerce", utc=False)
            wait_signal = "eligible_wait"
        elif queue_col:
            eligible_ts = pd.to_datetime(work[queue_col], errors="coerce", utc=False)
            wait_signal = "queue_wait"

        if eligible_ts is not None:
            wait_sec = (work["__start_ts"] - eligible_ts).dt.total_seconds().clip(lower=0)
        else:
            wait_sec = pd.Series([float("nan")] * len(work), index=work.index)

        duration_sec = pd.Series([float("nan")] * len(work), index=work.index)
        if duration_col:
            duration_sec = pd.to_numeric(work[duration_col], errors="coerce")
            if float(duration_sec.notna().mean()) < 0.5:
                duration_td = pd.to_timedelta(work[duration_col], errors="coerce").dt.total_seconds()
                duration_sec = duration_sec.where(duration_sec.notna(), duration_td)
        elif end_col:
            end_ts = pd.to_datetime(work[end_col], errors="coerce", utc=False)
            duration_sec = (end_ts - work["__start_ts"]).dt.total_seconds().clip(lower=0)

        if wait_sec.notna().sum() == 0 and duration_sec.notna().sum() > 0:
            wait_sec = duration_sec
            wait_signal = "duration_proxy"

        work["__wait_sec"] = pd.to_numeric(wait_sec, errors="coerce").fillna(0).clip(lower=0)
        work["__duration_sec"] = pd.to_numeric(duration_sec, errors="coerce").fillna(0).clip(lower=0)
        if not wait_signal:
            wait_signal = "none"

        work = work.loc[
            work["__start_ts"].notna()
            & (~work["process_norm"].isin(INVALID_STRINGS))
            & (~work["server_norm"].isin(INVALID_STRINGS))
        ].copy()
        if work.empty or float(work["__wait_sec"].sum()) <= 0.0:
            return PluginResult(
                "ok",
                "Process/server lever model not applicable: no usable wait signal",
                {
                    "rows": int(df.shape[0]),
                    "usable_rows": int(work.shape[0]),
                    "not_applicable_reason": "no_wait_signal",
                },
                [
                    {
                        "kind": "process_server_lever",
                        "decision": "not_applicable",
                        "measurement_type": "not_applicable",
                        "reason_code": "NO_WAIT_SIGNAL",
                        "reason": "Unable to infer queue/eligible/duration based wait signal.",
                    }
                ],
                [],
                None,
            )

        close_start_day = int(ctx.settings.get("close_cycle_start_day", 20))
        close_end_day = int(ctx.settings.get("close_cycle_end_day", 5))
        close_static_mask, close_dynamic_mask, used_dynamic_window, close_windows = resolve_close_cycle_masks(
            work["__start_ts"],
            ctx.run_dir,
            close_start_day=close_start_day,
            close_end_day=close_end_day,
        )
        if close_static_mask is None:
            days = work["__start_ts"].dt.day
            if close_start_day <= close_end_day:
                close_static_mask = (days >= close_start_day) & (days <= close_end_day)
            else:
                close_static_mask = (days >= close_start_day) | (days <= close_end_day)
        if close_dynamic_mask is None:
            close_dynamic_mask = close_static_mask
            used_dynamic_window = False

        acct_windows = load_accounting_windows_from_run(ctx.run_dir)
        if not acct_windows:
            roll_day = int(ctx.settings.get("roll_day_fallback", 5))
            acct_windows = infer_accounting_windows_from_timestamps(
                work["__start_ts"], roll_day=roll_day
            )
        acct_ranges = window_ranges(acct_windows, kind="accounting_month")
        accounting_month_mask = _mask_from_ranges(work["__start_ts"], acct_ranges)
        if not accounting_month_mask.any():
            accounting_month_mask = pd.Series(True, index=work.index)

        window_masks: dict[str, pd.Series] = {
            "accounting_month": accounting_month_mask.fillna(False),
            "close_static": close_static_mask.fillna(False),
            "close_dynamic": close_dynamic_mask.fillna(False),
        }

        stats_by_window: dict[str, pd.DataFrame] = {}
        for window_name, mask in window_masks.items():
            frame = work.loc[mask].copy()
            if frame.empty:
                stats_by_window[window_name] = pd.DataFrame(
                    columns=["process_norm", "wait_sec", "rows", "server_count"]
                )
                continue
            grouped = (
                frame.groupby("process_norm")
                .agg(
                    wait_sec=("__wait_sec", "sum"),
                    rows=("process_norm", "size"),
                    server_count=("server_norm", "nunique"),
                )
                .reset_index()
            )
            stats_by_window[window_name] = grouped

        close_dynamic_frame = work.loc[window_masks["close_dynamic"]].copy()
        if close_dynamic_frame.empty:
            close_dynamic_frame = work.loc[window_masks["accounting_month"]].copy()

        min_process_runs = int(ctx.settings.get("min_process_runs", 25))
        max_processes = int(ctx.settings.get("max_processes", 10))
        min_modeled_delta_hours = float(ctx.settings.get("min_modeled_delta_hours", 0.05))
        min_assignment_servers = int(ctx.settings.get("min_assignment_servers", 2))
        assignment_benefit_factor = float(ctx.settings.get("assignment_benefit_factor", 1.0))
        assignment_penalty_scale = float(ctx.settings.get("assignment_penalty_scale", 0.35))
        max_examples = int(ctx.settings.get("max_examples", 25))

        close_total_wait_sec = float(close_dynamic_frame["__wait_sec"].sum())
        close_candidates = (
            close_dynamic_frame.groupby("process_norm")
            .agg(
                close_wait_sec=("__wait_sec", "sum"),
                close_rows=("process_norm", "size"),
                server_count=("server_norm", "nunique"),
            )
            .reset_index()
            .sort_values(["close_wait_sec", "close_rows"], ascending=[False, False])
        )
        close_candidates = close_candidates.loc[
            close_candidates["close_rows"] >= min_process_runs
        ].head(max_processes)

        window_wait_by_process: dict[str, dict[str, float]] = {
            "accounting_month": {},
            "close_static": {},
            "close_dynamic": {},
        }
        for window_name, grouped in stats_by_window.items():
            for row in grouped.itertuples(index=False):
                window_wait_by_process[window_name][str(row.process_norm)] = float(
                    row.wait_sec
                )

        findings: list[dict[str, Any]] = []

        for row in close_candidates.itertuples(index=False):
            process = str(row.process_norm)
            close_wait_sec = float(row.close_wait_sec)
            if close_wait_sec <= 0.0:
                continue
            server_count = max(1, int(row.server_count))
            process_close = close_dynamic_frame.loc[
                close_dynamic_frame["process_norm"] == process
            ].copy()
            if process_close.empty:
                continue

            per_server = (
                process_close.groupby("server_norm")
                .agg(
                    wait_sec=("__wait_sec", "sum"),
                    rows=("process_norm", "size"),
                    duration_sec=("__duration_sec", "sum"),
                )
                .reset_index()
                .sort_values(["rows", "wait_sec"], ascending=[False, False])
            )
            top_servers = per_server["server_norm"].tolist()[:3]

            # Lever A: +1 capacity for this process/server pool.
            add_server_ratio = 1.0 / float(server_count + 1)
            delta_close_add_server_sec = close_wait_sec * add_server_ratio
            if (delta_close_add_server_sec / 3600.0) >= min_modeled_delta_hours:
                delta_acct_sec = (
                    window_wait_by_process["accounting_month"].get(process, 0.0)
                    * add_server_ratio
                )
                delta_static_sec = (
                    window_wait_by_process["close_static"].get(process, 0.0)
                    * add_server_ratio
                )
                delta_close_sec = (
                    window_wait_by_process["close_dynamic"].get(process, 0.0)
                    * add_server_ratio
                )
                _, eff_pct_acct = _window_metrics(
                    window_wait_by_process["accounting_month"].get(process, 0.0),
                    delta_acct_sec,
                )
                _, eff_pct_static = _window_metrics(
                    window_wait_by_process["close_static"].get(process, 0.0),
                    delta_static_sec,
                )
                delta_close_h, eff_pct_close = _window_metrics(
                    window_wait_by_process["close_dynamic"].get(process, 0.0),
                    delta_close_sec,
                )
                contention_pct_close = (
                    (delta_close_sec / close_total_wait_sec) * 100.0
                    if close_total_wait_sec > 0
                    else 0.0
                )
                findings.append(
                    {
                        "kind": "actionable_ops_lever",
                        "measurement_type": "modeled",
                        "decision": "actionable",
                        "title": f"Add one server lane for process {process}",
                        "action_type": "add_server",
                        "action": "add_server",
                        "process_norm": process,
                        "process": process,
                        "target_process_ids": [process],
                        "recommendation": (
                            f"Add one additional server lane for {process} during close windows. "
                            f"Modeled close-window gain is about {delta_close_h:.2f} hours "
                            f"({eff_pct_close:.2f}% for this process)."
                        ),
                        "expected_delta_seconds": float(delta_close_sec),
                        "modeled_delta_hours": float(delta_close_h),
                        "delta_hours_accounting_month": float(delta_acct_sec / 3600.0),
                        "delta_hours_close_static": float(delta_static_sec / 3600.0),
                        "delta_hours_close_dynamic": float(delta_close_h),
                        "efficiency_gain_pct_accounting_month": float(eff_pct_acct),
                        "efficiency_gain_pct_close_static": float(eff_pct_static),
                        "efficiency_gain_pct_close_dynamic": float(eff_pct_close),
                        "modeled_contention_reduction_pct_close": float(contention_pct_close),
                        "modeled_benefit_hours_close_dynamic": float(delta_close_h),
                        "modeled_cost_hours_close_dynamic": 0.0,
                        "modeled_net_hours_close_dynamic": float(delta_close_h),
                        "modeled_assumptions": [
                            "Process wait scales inversely with available server lanes.",
                            "Only server capacity is changed; process logic remains unchanged.",
                        ],
                        "evidence": {
                            "window_scope": "accounting_month/close_static/close_dynamic",
                            "server_count_baseline": int(server_count),
                            "server_count_modeled": int(server_count + 1),
                            "top_servers": top_servers,
                            "close_rows": int(row.close_rows),
                            "wait_signal": wait_signal,
                            "selected_row_ids": [int(i) for i in process_close.index.tolist()[:max_examples]],
                        },
                    }
                )

            # Lever B: assignment isolation to a single server lane.
            if int(per_server.shape[0]) < min_assignment_servers:
                continue
            pinned_row = per_server.iloc[0]
            pinned_server = str(pinned_row["server_norm"])
            rows_total = int(per_server["rows"].sum())
            rows_pinned = int(pinned_row["rows"])
            rows_to_move = max(0, rows_total - rows_pinned)
            wait_total = float(per_server["wait_sec"].sum())
            wait_to_move = max(0.0, wait_total - float(pinned_row["wait_sec"]))
            if rows_to_move <= 0 or wait_to_move <= 0.0:
                continue

            concentration_ratio = rows_to_move / float(max(rows_pinned, 1))
            penalty_ratio = max(0.0, min(0.95, concentration_ratio * assignment_penalty_scale))
            benefit_close_sec = wait_to_move * assignment_benefit_factor
            cost_close_sec = benefit_close_sec * penalty_ratio
            net_close_sec = max(0.0, benefit_close_sec - cost_close_sec)
            if (net_close_sec / 3600.0) < min_modeled_delta_hours:
                continue

            basis_close = window_wait_by_process["close_dynamic"].get(process, 0.0)
            if basis_close <= 0.0:
                continue
            benefit_ratio = min(1.0, benefit_close_sec / basis_close)
            cost_ratio = min(1.0, cost_close_sec / basis_close)
            net_ratio = max(0.0, benefit_ratio - cost_ratio)

            acct_basis = window_wait_by_process["accounting_month"].get(process, 0.0)
            static_basis = window_wait_by_process["close_static"].get(process, 0.0)
            close_basis = window_wait_by_process["close_dynamic"].get(process, 0.0)
            delta_acct_sec = acct_basis * net_ratio
            delta_static_sec = static_basis * net_ratio
            delta_close_sec = close_basis * net_ratio
            benefit_close_h = benefit_close_sec / 3600.0
            cost_close_h = cost_close_sec / 3600.0
            net_close_h = delta_close_sec / 3600.0
            _, eff_pct_acct = _window_metrics(acct_basis, delta_acct_sec)
            _, eff_pct_static = _window_metrics(static_basis, delta_static_sec)
            _, eff_pct_close = _window_metrics(close_basis, delta_close_sec)
            contention_pct_close = (
                (delta_close_sec / close_total_wait_sec) * 100.0
                if close_total_wait_sec > 0.0
                else 0.0
            )
            findings.append(
                {
                    "kind": "actionable_ops_lever",
                    "measurement_type": "modeled",
                    "decision": "actionable",
                    "title": f"Assign process {process} to a dedicated server lane",
                    "action_type": "tune_schedule",
                    "action": "tune_schedule",
                    "process_norm": process,
                    "process": process,
                    "target_process_ids": [process],
                    "recommendation": (
                        f"During close windows, pin {process} to server {pinned_server} and keep other lanes "
                        "available for close-critical work. "
                        f"Modeled benefit is {benefit_close_h:.2f}h contention relief, "
                        f"modeled cost is {cost_close_h:.2f}h extra wait on the pinned lane, "
                        f"net close-window gain is {net_close_h:.2f}h."
                    ),
                    "expected_delta_seconds": float(delta_close_sec),
                    "modeled_delta_hours": float(net_close_h),
                    "delta_hours_accounting_month": float(delta_acct_sec / 3600.0),
                    "delta_hours_close_static": float(delta_static_sec / 3600.0),
                    "delta_hours_close_dynamic": float(net_close_h),
                    "efficiency_gain_pct_accounting_month": float(eff_pct_acct),
                    "efficiency_gain_pct_close_static": float(eff_pct_static),
                    "efficiency_gain_pct_close_dynamic": float(eff_pct_close),
                    "modeled_contention_reduction_pct_close": float(contention_pct_close),
                    "modeled_benefit_hours_close_dynamic": float(benefit_close_h),
                    "modeled_cost_hours_close_dynamic": float(cost_close_h),
                    "modeled_net_hours_close_dynamic": float(net_close_h),
                    "modeled_assumptions": [
                        "Assignment isolates this process to one lane during close windows.",
                        "Benefit equals removed cross-lane wait; cost scales with concentration on pinned lane.",
                        "No process-code changes are assumed.",
                    ],
                    "evidence": {
                        "window_scope": "accounting_month/close_static/close_dynamic",
                        "pinned_server": pinned_server,
                        "rows_total": int(rows_total),
                        "rows_pinned_baseline": int(rows_pinned),
                        "rows_to_move": int(rows_to_move),
                        "benefit_ratio": float(benefit_ratio),
                        "cost_ratio": float(cost_ratio),
                        "net_ratio": float(net_ratio),
                        "wait_signal": wait_signal,
                        "selected_row_ids": [int(i) for i in process_close.index.tolist()[:max_examples]],
                    },
                }
            )

        if not findings:
            findings.append(
                {
                    "kind": "process_server_lever",
                    "decision": "not_applicable",
                    "measurement_type": "not_applicable",
                    "reason_code": "NO_ACTIONABLE_PROCESS_SERVER_LEVER",
                    "reason": (
                        "No process met minimum runs/benefit thresholds for +1 capacity or assignment tradeoff."
                    ),
                    "min_process_runs": min_process_runs,
                    "min_modeled_delta_hours": min_modeled_delta_hours,
                }
            )

        metrics = {
            "rows": int(work.shape[0]),
            "candidate_processes": int(close_candidates.shape[0]),
            "actionable_findings": int(
                sum(1 for f in findings if str(f.get("decision") or "") == "actionable")
            ),
            "close_total_wait_hours": float(close_total_wait_sec / 3600.0),
            "wait_signal": wait_signal,
            "used_dynamic_close_window": bool(used_dynamic_window),
            "close_window_source_count": int(len(close_windows)),
        }

        artifacts_dir = ctx.artifacts_dir("analysis_process_server_levers_v1")
        summary_path = artifacts_dir / "process_server_levers.json"
        write_json(
            summary_path,
            {
                "columns": {
                    "process": process_col,
                    "server": server_col,
                    "queue": queue_col,
                    "eligible": eligible_col,
                    "start": start_col,
                    "end": end_col,
                    "duration": duration_col,
                },
                "metrics": metrics,
                "findings_count": len(findings),
            },
        )
        artifacts = [
            PluginArtifact(
                path=str(summary_path.relative_to(ctx.run_dir)),
                type="json",
                description="Dynamic process/server lever modeling summary",
            )
        ]

        return PluginResult(
            "ok",
            "Modeled process-server +1 and assignment levers",
            metrics,
            findings,
            artifacts,
            None,
        )
