from __future__ import annotations

import csv
import math
from collections import Counter
from typing import Iterable

import pandas as pd

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not headers:
        return ""
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


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


def _candidate_columns(
    columns: list[str],
    role_by_name: dict[str, str],
    roles: set[str],
    patterns: list[str],
    lower_names: dict[str, str],
    exclude: set[str],
) -> list[str]:
    candidates: list[str] = []
    for col in columns:
        if col in exclude:
            continue
        if role_by_name.get(col) in roles:
            candidates.append(col)
    for col in columns:
        if col in exclude or col in candidates:
            continue
        name = lower_names[col]
        if any(pattern in name for pattern in patterns):
            candidates.append(col)
    return candidates


def _score_process_column(name: str, series: pd.Series) -> float:
    score = 0.0
    lower_name = name.lower()
    if lower_name in {"process", "process_id"}:
        score += 3.0
    if lower_name.endswith("_id") or lower_name.endswith("id"):
        score += 1.5
    for token in (
        "queue",
        "status",
        "step",
        "parent",
        "child",
        "hold",
        "lock",
        "schedule",
        "master",
        "dep",
        "ext",
        "attempt",
        "priority",
    ):
        if token in lower_name:
            score -= 2.0

    sample = series.dropna()
    if sample.empty:
        return score - 5.0
    if sample.shape[0] > 5000:
        sample = sample.sample(5000, random_state=0)

    if pd.api.types.is_numeric_dtype(sample):
        score -= 1.5
    else:
        score += 1.5

    sample_str = sample.astype(str).str.strip()
    if not pd.api.types.is_numeric_dtype(sample):
        numeric_like = sample_str.str.match(r"^\\d+(\\.\\d+)?$").mean()
        if numeric_like > 0.8:
            score -= 2.0

    unique_ratio = sample.nunique(dropna=True) / max(1, sample.shape[0])
    score += (1.0 - unique_ratio) * 4.0
    if unique_ratio > 0.9:
        score -= 2.0

    lengths = sample_str.str.len()
    median_len = float(lengths.median()) if not lengths.empty else 0.0
    if 3 <= median_len <= 20:
        score += 0.5
    elif median_len > 40:
        score -= 0.5

    return score


def _choose_best_process_column(
    candidates: Iterable[str], df: pd.DataFrame
) -> str | None:
    candidates = list(candidates)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    scored = []
    for col in candidates:
        scored.append((_score_process_column(str(col), df[col]), col))
    scored.sort(reverse=True, key=lambda item: item[0])
    return scored[0][1]


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

        preferred_process = ctx.settings.get("process_column")
        process_candidates = _candidate_columns(
            columns,
            role_by_name,
            {"process", "activity", "event", "step", "task", "action"},
            ["process", "activity", "event", "step", "task", "action", "job"],
            lower_names,
            used,
        )
        process_col = None
        if preferred_process and preferred_process in columns:
            process_col = preferred_process
        else:
            process_col = _choose_best_process_column(process_candidates, df)
        if process_col:
            used.add(process_col)

        timestamp_col = _pick_column(
            ctx.settings.get("timestamp_column"),
            columns,
            role_by_name,
            {"timestamp"},
            ["timestamp", "time", "date"],
            lower_names,
            used,
        )
        if timestamp_col:
            used.add(timestamp_col)

        start_col = _pick_column(
            ctx.settings.get("start_column"),
            columns,
            role_by_name,
            {"start"},
            ["start", "begin", "queued", "enqueue"],
            lower_names,
            used,
        )
        if start_col:
            used.add(start_col)

        end_col = _pick_column(
            ctx.settings.get("end_column"),
            columns,
            role_by_name,
            {"end", "finish", "complete"},
            ["end", "finish", "complete", "stop", "dequeue"],
            lower_names,
            used,
        )
        if end_col:
            used.add(end_col)

        duration_col = _pick_column(
            ctx.settings.get("duration_column"),
            columns,
            role_by_name,
            {"duration", "latency", "elapsed", "runtime"},
            ["duration", "elapsed", "latency", "runtime", "seconds", "secs", "ms"],
            lower_names,
            used,
        )
        if duration_col:
            used.add(duration_col)

        server_col = _pick_column(
            ctx.settings.get("server_column"),
            columns,
            role_by_name,
            {"server", "host", "node", "instance"},
            ["server", "host", "node", "instance", "machine"],
            lower_names,
            used,
        )
        if server_col:
            used.add(server_col)

        param_col = _pick_column(
            ctx.settings.get("param_column"),
            columns,
            role_by_name,
            {"parameter", "params", "config", "meta"},
            ["param", "parameter", "params", "config", "meta"],
            lower_names,
            used,
        )

        if not process_col:
            return PluginResult(
                "skipped", "No process/activity column detected", {}, [], [], None
            )

        base_timestamp_col = timestamp_col or start_col or end_col
        if not base_timestamp_col:
            return PluginResult(
                "skipped", "No timestamp column detected", {}, [], [], None
            )

        work = df.copy()
        selected_cols: list[str] = []
        for col in [
            process_col,
            base_timestamp_col,
            duration_col,
            start_col,
            end_col,
            server_col,
            param_col,
        ]:
            if col and col in work.columns and col not in selected_cols:
                selected_cols.append(col)
        work = work.loc[:, selected_cols]

        work["__timestamp"] = pd.to_datetime(
            work[base_timestamp_col], errors="coerce", utc=False
        )
        work = work.loc[work["__timestamp"].notna()].copy()
        if work.empty:
            return PluginResult(
                "skipped", "No valid timestamps found", {}, [], [], None
            )

        duration_label = duration_col
        if duration_col and duration_col in work.columns:
            duration = pd.to_numeric(work[duration_col], errors="coerce")
            if duration.isna().all():
                duration = pd.to_timedelta(
                    work[duration_col], errors="coerce"
                ).dt.total_seconds()
        elif start_col and end_col and start_col in work.columns and end_col in work.columns:
            start_ts = pd.to_datetime(work[start_col], errors="coerce", utc=False)
            end_ts = pd.to_datetime(work[end_col], errors="coerce", utc=False)
            duration = (end_ts - start_ts).dt.total_seconds()
            duration_label = f"{start_col}->{end_col}"
        else:
            return PluginResult(
                "skipped",
                "No duration data available",
                {},
                [],
                [],
                None,
            )

        work["__duration"] = duration
        work = work.loc[work["__duration"].notna() & (work["__duration"] > 0)].copy()
        if work.empty:
            return PluginResult("skipped", "No valid durations found", {}, [], [], None)

        work["__process"] = work[process_col].astype(str).str.strip()
        work["__process_norm"] = work["__process"].str.lower()
        invalid = {"", "nan", "none", "null"}
        work = work.loc[~work["__process_norm"].isin(invalid)].copy()
        if work.empty:
            return PluginResult("skipped", "No valid process values", {}, [], [], None)

        close_start = int(ctx.settings.get("close_cycle_start_day", 20))
        close_end = int(ctx.settings.get("close_cycle_end_day", 5))
        min_close_count = int(ctx.settings.get("min_close_count", 200))
        min_days = int(ctx.settings.get("min_days", 20))
        min_share_delta = float(ctx.settings.get("min_share_delta", 0.002))
        min_slowdown_ratio = float(ctx.settings.get("min_slowdown_ratio", 1.05))
        p_value_threshold = float(ctx.settings.get("p_value_threshold", 0.1))
        max_examples = int(ctx.settings.get("max_examples", 25))
        min_open_count = int(ctx.settings.get("min_open_count", 200))
        min_open_days = int(ctx.settings.get("min_open_days", 10))
        min_open_share = float(ctx.settings.get("min_open_share", 0.002))
        min_open_ratio = float(ctx.settings.get("min_open_ratio", 0.02))
        suppress_list = ctx.settings.get("suppress_processes") or []
        if isinstance(suppress_list, str):
            suppress_list = [
                item.strip()
                for item in suppress_list.split(",")
                if item.strip()
            ]
        suppress_set = {str(item).strip().lower() for item in suppress_list if str(item).strip()}
        suppress_regex = ctx.settings.get("suppress_process_regex")
        suppress_pattern = None
        if isinstance(suppress_regex, str) and suppress_regex.strip():
            import re

            suppress_pattern = re.compile(suppress_regex, re.IGNORECASE)

        work["__day"] = work["__timestamp"].dt.day
        work["__date"] = work["__timestamp"].dt.date
        if close_start <= close_end:
            work["__close"] = (work["__day"] >= close_start) & (
                work["__day"] <= close_end
            )
        else:
            work["__close"] = (work["__day"] >= close_start) | (
                work["__day"] <= close_end
            )

        close = work.loc[work["__close"]].copy()
        open_rows = work.loc[~work["__close"]].copy()

        def emit_results(
            summary: dict[str, object],
            evaluated: list[dict[str, object]],
            candidates: list[dict[str, object]],
            findings: list[dict[str, object]],
            message: str,
        ) -> PluginResult:
            artifacts = []
            artifacts_dir = ctx.artifacts_dir("analysis_close_cycle_uplift")
            out_path = artifacts_dir / "results.json"
            write_json(
                out_path,
                {
                    "summary": summary,
                    "candidates": candidates,
                    "evaluated": evaluated,
                },
            )
            artifacts.append(
                PluginArtifact(
                    path=str(out_path.relative_to(ctx.run_dir)),
                    type="json",
                    description="Close cycle share-shift candidates",
                )
            )

            csv_path = artifacts_dir / "results.csv"
            csv_headers = [
                "process",
                "close_count",
                "open_count",
                "open_days",
                "open_ratio",
                "close_days",
                "close_share",
                "open_share",
                "share_delta",
                "z_score",
                "p_value",
                "slowdown_ratio",
                "suppressed",
                "suppression_reason",
                "reason",
            ]
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(csv_headers)
                for entry in evaluated:
                    writer.writerow(
                        [
                            entry.get("process"),
                            entry.get("close_count"),
                            entry.get("open_count"),
                            entry.get("open_days"),
                            entry.get("open_ratio"),
                            entry.get("close_days"),
                            entry.get("close_share"),
                            entry.get("open_share"),
                            entry.get("share_delta"),
                            entry.get("z_score"),
                            entry.get("p_value"),
                            entry.get("slowdown_ratio"),
                            entry.get("suppressed"),
                            entry.get("suppression_reason"),
                            entry.get("reason"),
                        ]
                    )
            artifacts.append(
                PluginArtifact(
                    path=str(csv_path.relative_to(ctx.run_dir)),
                    type="csv",
                    description="Share-shift detail table",
                )
            )

            detected = [
                entry for entry in evaluated if entry.get("passed") and not entry.get("suppressed")
            ]
            suppressed = [entry for entry in evaluated if entry.get("suppressed")]
            near = [
                entry for entry in evaluated if (not entry.get("passed")) and not entry.get("suppressed")
            ]
            detected = sorted(detected, key=lambda r: float(r.get("share_delta", 0)), reverse=True)
            near = sorted(near, key=lambda r: float(r.get("share_delta", 0)), reverse=True)[:10]
            suppressed = sorted(
                suppressed, key=lambda r: float(r.get("share_delta", 0)), reverse=True
            )[:10]

            md_lines = [
                "# Close-cycle share shift",
                "",
                "Summary:",
                f"- close_cycle_start_day: {summary.get('close_cycle_start_day')}",
                f"- close_cycle_end_day: {summary.get('close_cycle_end_day')}",
                f"- min_close_count: {summary.get('min_close_count')}",
                f"- min_days: {summary.get('min_days')}",
                f"- min_share_delta: {summary.get('min_share_delta')}",
                f"- min_slowdown_ratio: {summary.get('min_slowdown_ratio')}",
                f"- p_value_threshold: {summary.get('p_value_threshold')}",
                f"- min_open_count: {summary.get('min_open_count')}",
                f"- min_open_days: {summary.get('min_open_days')}",
                f"- min_open_share: {summary.get('min_open_share')}",
                f"- min_open_ratio: {summary.get('min_open_ratio')}",
                f"- suppress_processes: {summary.get('suppress_processes')}",
                f"- suppress_process_regex: {summary.get('suppress_process_regex')}",
                f"- median_close: {summary.get('median_close')}",
                f"- median_open: {summary.get('median_open')}",
                f"- slowdown_ratio: {summary.get('slowdown_ratio')}",
                "",
                "Detected:",
            ]

            headers = [
                "process",
                "close_share",
                "open_share",
                "delta",
                "z",
                "p",
                "close_count",
                "open_count",
            ]
            detected_rows = [
                [
                    str(row.get("process", "")),
                    _fmt_float(row.get("close_share")),
                    _fmt_float(row.get("open_share")),
                    _fmt_float(row.get("share_delta")),
                    _fmt_float(row.get("z_score"), 2),
                    _fmt_float(row.get("p_value"), 6),
                    str(row.get("close_count")),
                    str(row.get("open_count")),
                ]
                for row in detected
            ]
            md_lines.append(_markdown_table(headers, detected_rows) if detected_rows else "_None_")

            md_lines.extend(["", "Detected reason summary:"])
            if detected:
                slowdown_ratio = summary.get("slowdown_ratio")
                min_slowdown_ratio = summary.get("min_slowdown_ratio")
                min_open_ratio = summary.get("min_open_ratio")
                min_open_count = summary.get("min_open_count")
                min_open_days = summary.get("min_open_days")
                min_close_count = summary.get("min_close_count")
                min_close_days = summary.get("min_days")
                for row in detected:
                    md_lines.append(
                        "- "
                        + f"{row.get('process')}: "
                        + f"close_share { _fmt_float(row.get('close_share')) } "
                        + f"vs open_share { _fmt_float(row.get('open_share')) } "
                        + f"(delta { _fmt_float(row.get('share_delta')) }); "
                        + f"z { _fmt_float(row.get('z_score'), 2) }, "
                        + f"p { _fmt_float(row.get('p_value'), 6) }; "
                        + f"open_ratio { _fmt_float(row.get('open_ratio'), 3) } "
                        + f">= { _fmt_float(min_open_ratio, 3) }; "
                        + f"open_count {row.get('open_count')} >= {min_open_count}, "
                        + f"open_days {row.get('open_days')} >= {min_open_days}; "
                        + f"close_count {row.get('close_count')} >= {min_close_count}, "
                        + f"close_days {row.get('close_days')} >= {min_close_days}; "
                        + f"global_slowdown_ratio { _fmt_float(slowdown_ratio, 3) } "
                        + f">= { _fmt_float(min_slowdown_ratio, 3) }"
                    )
            else:
                md_lines.append("_None_")

            md_lines.extend(["", "Suppressed (expected close-only or manual):"])
            suppressed_rows = [
                [
                    str(row.get("process", "")),
                    _fmt_float(row.get("close_share")),
                    _fmt_float(row.get("open_share")),
                    _fmt_float(row.get("share_delta")),
                    _fmt_float(row.get("z_score"), 2),
                    _fmt_float(row.get("p_value"), 6),
                    str(row.get("close_count")),
                    str(row.get("open_count")),
                    str(row.get("suppression_reason", "")),
                ]
                for row in suppressed
            ]
            suppressed_headers = headers + ["suppression_reason"]
            md_lines.append(
                _markdown_table(suppressed_headers, suppressed_rows)
                if suppressed_rows
                else "_None_"
            )

            md_lines.extend(["", "Near misses (top share deltas):"])
            near_rows = [
                [
                    str(row.get("process", "")),
                    _fmt_float(row.get("close_share")),
                    _fmt_float(row.get("open_share")),
                    _fmt_float(row.get("share_delta")),
                    _fmt_float(row.get("z_score"), 2),
                    _fmt_float(row.get("p_value"), 6),
                    str(row.get("close_count")),
                    str(row.get("open_count")),
                    str(row.get("reason", "")),
                ]
                for row in near
            ]
            near_headers = headers + ["reason"]
            md_lines.append(_markdown_table(near_headers, near_rows) if near_rows else "_None_")

            md_path = artifacts_dir / "results.md"
            md_path.write_text("\n".join(md_lines), encoding="utf-8")
            artifacts.append(
                PluginArtifact(
                    path=str(md_path.relative_to(ctx.run_dir)),
                    type="markdown",
                    description="Share-shift summary",
                )
            )

            metrics = dict(summary)
            metrics["candidates"] = len(findings)
            return PluginResult("ok", message, metrics, findings, artifacts, None)

        summary = {
            "process_column": process_col,
            "timestamp_column": base_timestamp_col,
            "duration_column": duration_label,
            "server_column": server_col,
            "param_column": param_col,
            "close_cycle_start_day": close_start,
            "close_cycle_end_day": close_end,
            "min_close_count": min_close_count,
            "min_days": min_days,
            "min_share_delta": min_share_delta,
            "min_slowdown_ratio": min_slowdown_ratio,
            "p_value_threshold": p_value_threshold,
            "min_open_count": min_open_count,
            "min_open_days": min_open_days,
            "min_open_share": min_open_share,
            "min_open_ratio": min_open_ratio,
            "suppress_processes": sorted(suppress_set),
            "suppress_process_regex": suppress_regex,
        }

        evaluated: list[dict[str, object]] = []
        candidates: list[dict[str, object]] = []
        findings: list[dict[str, object]] = []

        if close.empty or open_rows.empty:
            summary.update(
                {
                    "median_close": None,
                    "median_open": None,
                    "slowdown_ratio": None,
                }
            )
            return emit_results(
                summary,
                evaluated,
                candidates,
                findings,
                "No close-cycle rows found",
            )

        median_close = float(close["__duration"].median())
        median_open = float(open_rows["__duration"].median())
        slowdown_ratio = None
        if median_open > 0:
            slowdown_ratio = float(median_close / median_open)
        summary.update(
            {
                "median_close": median_close,
                "median_open": median_open,
                "slowdown_ratio": slowdown_ratio,
            }
        )

        process_labels = (
            close.groupby("__process_norm")["__process"]
            .agg(lambda series: series.value_counts().index[0])
            .to_dict()
        )

        close_counts = close["__process_norm"].value_counts()
        open_counts = open_rows["__process_norm"].value_counts()
        total_close = int(close_counts.sum())
        total_open = int(open_counts.sum())

        close_days_by_process = (
            close.groupby("__process_norm")["__date"]
            .nunique()
            .to_dict()
        )
        open_days_by_process = (
            open_rows.groupby("__process_norm")["__date"]
            .nunique()
            .to_dict()
        )

        all_processes = sorted(set(close_counts.index).union(set(open_counts.index)))
        global_ok = slowdown_ratio is not None and slowdown_ratio >= min_slowdown_ratio

        for proc in all_processes:
            close_count = int(close_counts.get(proc, 0))
            open_count = int(open_counts.get(proc, 0))
            close_days_count = int(close_days_by_process.get(proc, 0))
            open_days_count = int(open_days_by_process.get(proc, 0))

            p_close = close_count / total_close if total_close else 0.0
            p_open = open_count / total_open if total_open else 0.0
            share_delta = p_close - p_open
            open_ratio = (
                open_count / (open_count + close_count)
                if (open_count + close_count) > 0
                else 0.0
            )

            pooled = (close_count + open_count) / (
                total_close + total_open
            ) if (total_close + total_open) else 0.0
            se = math.sqrt(
                pooled
                * (1.0 - pooled)
                * ((1.0 / total_close) + (1.0 / total_open))
            ) if pooled > 0 and total_close > 0 and total_open > 0 else 0.0
            z_score = (share_delta / se) if se > 0 else 0.0
            p_value = 0.5 * math.erfc(z_score / math.sqrt(2.0)) if se > 0 else 1.0

            reasons: list[str] = []
            if not global_ok:
                reasons.append("global_slowdown")
            if close_count < min_close_count:
                reasons.append("close_count")
            if close_days_count < min_days:
                reasons.append("close_days")
            if share_delta < min_share_delta:
                reasons.append("share_delta")
            if p_value > p_value_threshold:
                reasons.append("p_value")

            suppression_reasons: list[str] = []
            if proc in suppress_set:
                suppression_reasons.append("manual")
            process_label = process_labels.get(proc, proc)
            if suppress_pattern and suppress_pattern.search(process_label):
                suppression_reasons.append("manual_regex")
            if open_count < min_open_count:
                suppression_reasons.append("open_count")
            if open_days_count < min_open_days:
                suppression_reasons.append("open_days")
            if p_open < min_open_share:
                suppression_reasons.append("open_share")
            if open_ratio < min_open_ratio:
                suppression_reasons.append("open_ratio")

            suppressed = len(suppression_reasons) > 0
            passed = len(reasons) == 0 and not suppressed
            reason_text = "ok" if len(reasons) == 0 else ",".join(reasons)
            suppression_text = (
                ""
                if not suppressed
                else ",".join(sorted(set(suppression_reasons)))
            )

            evaluated.append(
                {
                    "process": process_label,
                    "process_norm": proc,
                    "close_count": close_count,
                    "open_count": open_count,
                    "open_days": open_days_count,
                    "open_ratio": float(open_ratio),
                    "close_days": close_days_count,
                    "close_share": float(p_close),
                    "open_share": float(p_open),
                    "share_delta": float(share_delta),
                    "z_score": float(z_score),
                    "p_value": float(p_value),
                    "slowdown_ratio": slowdown_ratio,
                    "passed": passed,
                    "suppressed": suppressed,
                    "suppression_reason": suppression_text,
                    "reason": reason_text,
                }
            )

            if not passed:
                continue

            expected_close = p_open * total_close
            excess_close = close_count - expected_close

            server_list: list[str] = []
            server_count = 0
            if server_col and server_col in close.columns:
                servers = (
                    close.loc[close["__process_norm"] == proc, server_col]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )
                if not servers.empty:
                    counter = Counter(servers)
                    server_list = [name for name, _ in counter.most_common(5)]
                    server_count = len(counter)

            row_ids = []
            for idx in close.loc[close["__process_norm"] == proc].index.tolist():
                try:
                    row_ids.append(int(idx))
                except (TypeError, ValueError):
                    continue
            row_ids = row_ids[:max_examples]

            process_label = process_labels.get(proc, proc)
            columns = [process_col, base_timestamp_col]
            if duration_col and duration_col in work.columns:
                columns.append(duration_col)
            elif start_col and end_col:
                columns.extend([start_col, end_col])
            if server_col:
                columns.append(server_col)
            if param_col:
                columns.append(param_col)

            finding = {
                "kind": "close_cycle_share_shift",
                "process": process_label,
                "process_norm": proc,
                "close_count": close_count,
                "open_count": open_count,
                "close_cycle_days": close_days_count,
                "close_share": float(p_close),
                "open_share": float(p_open),
                "share_delta": float(share_delta),
                "expected_close": float(expected_close),
                "excess_close": float(excess_close),
                "z_score": float(z_score),
                "p_value": float(p_value),
                "median_close": float(median_close),
                "median_open": float(median_open),
                "slowdown_ratio": float(slowdown_ratio),
                "server_count": int(server_count),
                "servers": server_list,
                "columns": columns,
                "row_ids": row_ids,
                "query": f"process={process_label}",
            }
            findings.append(finding)
            candidates.append(
                {
                    "process": process_label,
                    "process_norm": proc,
                    "close_count": close_count,
                    "open_count": open_count,
                    "close_cycle_days": close_days_count,
                    "share_delta": float(share_delta),
                    "z_score": float(z_score),
                    "p_value": float(p_value),
                }
            )

        if not findings:
            return emit_results(
                summary,
                evaluated,
                candidates,
                findings,
                "No close-cycle share-shift candidates found",
            )

        return emit_results(
            summary,
            evaluated,
            candidates,
            findings,
            "Detected close-cycle share-shift candidates",
        )
