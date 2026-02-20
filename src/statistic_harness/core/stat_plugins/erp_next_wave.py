from __future__ import annotations

from typing import Any, Callable


import numpy as np
import pandas as pd

from statistic_harness.core.ideaspace_feature_extractor import coerce_datetime
from statistic_harness.core.stat_plugins import BudgetTimer, stable_id
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


def _basic_metrics(df: pd.DataFrame, sample_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows_seen": int(sample_meta.get("rows_seen", len(df))),
        "rows_used": int(sample_meta.get("rows_used", len(df))),
        "cols_used": int(len(df.columns)),
        "sampled": bool(sample_meta.get("sampled", False)),
    }


def _artifact(ctx, plugin_id: str, name: str, payload: Any) -> PluginArtifact:
    artifact_dir = ctx.artifacts_dir(plugin_id)
    path = artifact_dir / name
    write_json(path, payload)
    return PluginArtifact(
        path=str(path.relative_to(ctx.run_dir)),
        type="json",
        description=name,
    )


def _pick_col_by_name(df: pd.DataFrame, candidates: list[str], hints: tuple[str, ...]) -> str | None:
    for hint in hints:
        hint_s = str(hint).strip().lower()
        if not hint_s:
            continue
        for col in candidates:
            col_s = str(col)
            if hint_s in col_s.lower():
                if col in df.columns:
                    return str(col)
                if col_s in df.columns:
                    return col_s
    return None


def _pick_process_col(df: pd.DataFrame, inferred: dict[str, Any]) -> str | None:
    cat_cols = [str(c) for c in (inferred.get("categorical_columns") or []) if str(c) in df.columns]
    proc = _pick_col_by_name(df, cat_cols, ("process_id", "process", "proc", "activity"))
    if proc:
        return proc
    for col in df.columns:
        if "process" in str(col).lower():
            return str(col)
    return None


def _hold_time_attribution_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    proc_col = _pick_process_col(df, inferred)
    ts_cols = [str(c) for c in (inferred.get("timestamp_columns") or []) if str(c) in df.columns]
    queue_col = _pick_col_by_name(df, ts_cols, ("queue", "queued"))
    eligible_col = _pick_col_by_name(df, ts_cols, ("eligible",))
    start_col = _pick_col_by_name(df, ts_cols, ("start", "begin"))
    if not (proc_col and start_col and (queue_col or eligible_col)):
        return PluginResult(
            "ok",
            "Hold-time attribution unavailable: required columns missing",
            _basic_metrics(df, sample_meta),
            [
                {
                    "id": stable_id(f"{plugin_id}:missing_columns"),
                    "kind": "plugin_observation",
                    "severity": "info",
                    "confidence": 1.0,
                    "title": "Hold-time attribution unavailable for this dataset",
                    "what": "Required queue/eligible/start/process columns are not all present.",
                    "why": "Without those columns the plugin cannot separate hold time from eligible-wait time.",
                    "evidence": {
                        "metrics": {
                            "process_column": proc_col,
                            "queue_column": queue_col,
                            "eligible_column": eligible_col,
                            "start_column": start_col,
                        }
                    },
                    "measurement_type": "system_derived",
                    "reason_code": "MISSING_REQUIRED_COLUMNS",
                }
            ],
            [],
            None,
            debug={"gating_reason": "missing_columns", "proc_col": proc_col, "queue_col": queue_col, "eligible_col": eligible_col, "start_col": start_col},
        )

    start = coerce_datetime(df[start_col])
    base_queue = coerce_datetime(df[eligible_col] if eligible_col else df[queue_col])  # type: ignore[index]
    queue = coerce_datetime(df[queue_col]) if queue_col else base_queue
    eligible = coerce_datetime(df[eligible_col]) if eligible_col else base_queue
    ok = start.notna() & queue.notna() & eligible.notna()
    if ok.mean() < 0.7:
        return PluginResult(
            "degraded",
            "Low timestamp parse rate; hold attribution may be incomplete",
            _basic_metrics(df, sample_meta),
            [],
            [],
            None,
            debug={"parse_rate": float(ok.mean())},
        )

    hold_s = (eligible[ok] - queue[ok]).dt.total_seconds().clip(lower=0.0)
    ew_s = (start[ok] - eligible[ok]).dt.total_seconds().clip(lower=0.0)
    frame = pd.DataFrame(
        {
            "process": df.loc[ok, proc_col].astype(str),
            "hold_s": hold_s.to_numpy(dtype=float),
            "eligible_wait_s": ew_s.to_numpy(dtype=float),
        }
    )

    g = frame.groupby("process", dropna=False)
    summary = g.agg(
        runs=("process", "size"),
        hold_hours=("hold_s", lambda x: float(np.nansum(x.to_numpy(dtype=float)) / 3600.0)),
        eligible_wait_hours=("eligible_wait_s", lambda x: float(np.nansum(x.to_numpy(dtype=float)) / 3600.0)),
        hold_p95_s=("hold_s", lambda x: float(np.nanpercentile(x.to_numpy(dtype=float), 95)) if len(x) else 0.0),
        eligible_wait_p95_s=("eligible_wait_s", lambda x: float(np.nanpercentile(x.to_numpy(dtype=float), 95)) if len(x) else 0.0),
    ).reset_index()
    summary["total_hours"] = summary["hold_hours"] + summary["eligible_wait_hours"]
    summary["hold_share"] = summary["hold_hours"] / summary["total_hours"].replace(0.0, np.nan)
    summary = summary.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    min_hours = float(config.get("min_total_hours", 10.0))
    candidates = summary.loc[(summary["total_hours"] >= min_hours) & (summary["hold_share"] >= 0.5)]
    candidates = candidates.sort_values(["hold_share", "total_hours", "process"], ascending=[False, False, True]).head(10)

    findings: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        process = str(row["process"])
        findings.append(
            {
                "id": stable_id(f"{plugin_id}:{process}"),
                "kind": "hold_time_attribution",
                "severity": "warn",
                "confidence": 0.6,
                "title": f"Hold-time dominates wait for {process}",
                "what": "A large fraction of delay happens before the item becomes eligible to run (holds/dependencies), not just in the runnable queue.",
                "why": "Reducing hold time often yields larger improvements than adding capacity when holds dominate.",
                "evidence": {
                    "metrics": {
                        "process": process,
                        "runs": int(row["runs"]),
                        "hold_hours": float(row["hold_hours"]),
                        "eligible_wait_hours": float(row["eligible_wait_hours"]),
                        "hold_share": float(row["hold_share"]),
                        "hold_p95_s": float(row["hold_p95_s"]),
                    }
                },
                "recommendations": [
                    "Identify dependency gates/holds for this process and reduce their duration or frequency.",
                    "Check upstream prerequisites that delay eligibility (data availability, locks, approvals, batch windows).",
                ],
                "limitations": [
                    "Attribution uses inferred queue/eligible/start timestamps and may be incomplete if the schema differs.",
                ],
                "measurement_type": "measured",
            }
        )

    artifacts = [
        _artifact(ctx, plugin_id, "hold_time_attribution.json", {"rows": len(df), "top": candidates.to_dict(orient="records")})
    ]
    if findings:
        return PluginResult(
            "ok",
            "Computed hold vs eligible-wait attribution",
            _basic_metrics(df, sample_meta),
            findings,
            artifacts,
            None,
        )
    observation = {
        "id": stable_id(f"{plugin_id}:no_hold_dominant"),
        "kind": "plugin_observation",
        "severity": "info",
        "confidence": 1.0,
        "title": "No hold-dominant processes detected",
        "what": "No process crossed the configured hold-share and volume thresholds.",
        "why": "This dataset currently shows queue/eligible waits that are not dominated by pre-eligible hold delay.",
        "evidence": {"metrics": {"min_total_hours": min_hours, "candidate_count": int(len(candidates))}},
        "measurement_type": "measured",
        "reason_code": "NO_HOLD_DOMINANT_PROCESSES",
    }
    return PluginResult(
        "ok",
        "Computed hold vs eligible-wait attribution (no dominant hold candidates)",
        _basic_metrics(df, sample_meta),
        [observation],
        artifacts,
        None,
    )


def _retry_rate_hotspots_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    proc_col = _pick_process_col(df, inferred)
    num_cols = [str(c) for c in (inferred.get("numeric_columns") or []) if str(c) in df.columns]
    attempts_col = _pick_col_by_name(df, num_cols, ("attempt", "retry"))
    if not (proc_col and attempts_col):
        return PluginResult(
            "ok",
            "Retry-rate hotspot analysis unavailable: required columns missing",
            _basic_metrics(df, sample_meta),
            [
                {
                    "id": stable_id(f"{plugin_id}:missing_columns"),
                    "kind": "plugin_observation",
                    "severity": "info",
                    "confidence": 1.0,
                    "title": "Retry-rate hotspot analysis unavailable for this dataset",
                    "what": "The dataset does not expose a usable attempts/retry numeric column.",
                    "why": "Retry-rate estimation requires a per-run attempt count.",
                    "evidence": {"metrics": {"process_column": proc_col, "attempts_column": attempts_col}},
                    "measurement_type": "system_derived",
                    "reason_code": "MISSING_REQUIRED_COLUMNS",
                }
            ],
            [],
            None,
            debug={"gating_reason": "missing_columns", "proc_col": proc_col, "attempts_col": attempts_col},
        )

    attempts = pd.to_numeric(df[attempts_col], errors="coerce")
    ok = attempts.notna()
    if ok.mean() < 0.5:
        return PluginResult(
            "ok",
            "Retry-rate hotspot analysis unavailable: insufficient attempts coverage",
            _basic_metrics(df, sample_meta),
            [
                {
                    "id": stable_id(f"{plugin_id}:insufficient_attempts"),
                    "kind": "plugin_observation",
                    "severity": "info",
                    "confidence": 1.0,
                    "title": "Retry-rate hotspot analysis unavailable for this dataset",
                    "what": "Attempts values were missing for most rows.",
                    "why": "Retry-rate estimates would be unreliable with this coverage level.",
                    "evidence": {"metrics": {"attempts_non_null_rate": float(ok.mean())}},
                    "measurement_type": "system_derived",
                    "reason_code": "INSUFFICIENT_ATTEMPTS_DATA",
                }
            ],
            [],
            None,
            debug={"gating_reason": "insufficient_attempts"},
        )

    frame = pd.DataFrame({"process": df.loc[ok, proc_col].astype(str), "attempts": attempts.loc[ok].to_numpy(dtype=float)})
    g = frame.groupby("process", dropna=False)
    summary = g.agg(
        runs=("process", "size"),
        retry_rate=("attempts", lambda x: float((x.to_numpy(dtype=float) > 1).mean()) if len(x) else 0.0),
        p95_attempts=("attempts", lambda x: float(np.nanpercentile(x.to_numpy(dtype=float), 95)) if len(x) else 1.0),
    ).reset_index()
    min_runs = int(config.get("min_runs", 200))
    min_retry_rate = float(config.get("min_retry_rate", 0.05))
    candidates = summary.loc[(summary["runs"] >= min_runs) & (summary["retry_rate"] >= min_retry_rate)]
    candidates = candidates.sort_values(["retry_rate", "runs", "process"], ascending=[False, False, True]).head(10)

    findings: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        process = str(row["process"])
        findings.append(
            {
                "id": stable_id(f"{plugin_id}:{process}"),
                "kind": "retry_rate_hotspot",
                "severity": "warn",
                "confidence": 0.55,
                "title": f"High retry rate for {process}",
                "what": "This process frequently needs multiple execution attempts.",
                "why": "Retries amplify load and can create self-inflicted queue delays during close windows.",
                "evidence": {"metrics": {"process": process, "runs": int(row["runs"]), "retry_rate": float(row["retry_rate"]), "p95_attempts": float(row["p95_attempts"]), "attempts_col": attempts_col}},
                "recommendations": [
                    "Investigate the dominant failure mode that causes retries and fix it at the source.",
                    "If retries are expected, introduce deterministic backoff and dedupe to avoid burst retries.",
                ],
                "limitations": ["Requires an attempts/retry column with consistent semantics across the ERP."],
                "measurement_type": "measured",
            }
        )

    artifacts = [
        _artifact(ctx, plugin_id, "retry_rate_hotspots.json", {"top": candidates.to_dict(orient="records"), "attempts_col": attempts_col})
    ]
    if findings:
        return PluginResult(
            "ok",
            "Computed retry-rate hotspots",
            _basic_metrics(df, sample_meta),
            findings,
            artifacts,
            None,
        )
    observation = {
        "id": stable_id(f"{plugin_id}:no_retry_hotspots"),
        "kind": "plugin_observation",
        "severity": "info",
        "confidence": 1.0,
        "title": "No high-retry hotspots detected",
        "what": "No process crossed the configured retry-rate and minimum-run thresholds.",
        "why": "Current retries do not cluster strongly enough to justify a targeted hotspot action here.",
        "evidence": {
            "metrics": {
                "min_runs": min_runs,
                "min_retry_rate": min_retry_rate,
                "candidate_count": int(len(candidates)),
            }
        },
        "measurement_type": "measured",
        "reason_code": "NO_RETRY_HOTSPOTS",
    }
    return PluginResult(
        "ok",
        "Computed retry-rate hotspots (no hotspot candidates)",
        _basic_metrics(df, sample_meta),
        [observation],
        artifacts,
        None,
    )


def _dependency_critical_path_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    # Look for queue/process id + parent/dependency id columns.
    cat_cols = [str(c) for c in (inferred.get("categorical_columns") or []) if str(c) in df.columns]
    id_col = _pick_col_by_name(df, cat_cols, ("process_queue_id", "queue_id", "process_id", "id"))
    parent_col = _pick_col_by_name(df, cat_cols, ("parent_process_queue_id", "parent_id", "dep_process_queue_id", "dep_id", "depends"))
    if not (id_col and parent_col):
        return PluginResult(
            "skipped",
            "Missing required columns for dependency critical path",
            _basic_metrics(df, sample_meta),
            [],
            [],
            None,
            debug={"gating_reason": "missing_columns", "id_col": id_col, "parent_col": parent_col},
        )

    ids = df[id_col]
    parents = df[parent_col]
    mapping: dict[str, str | None] = {}
    for i, p in zip(ids.tolist(), parents.tolist(), strict=False):
        iv = str(i).strip()
        if not iv or iv.lower() in {"nan", "none"}:
            continue
        pv = str(p).strip()
        if not pv or pv.lower() in {"nan", "none"}:
            mapping[iv] = None
        else:
            mapping[iv] = pv

    max_depth = int(config.get("max_depth", 50))
    memo: dict[str, int] = {}
    visiting: set[str] = set()

    def depth(node: str) -> int:
        if node in memo:
            return memo[node]
        if node in visiting:
            # Cycle guard.
            memo[node] = 1
            return 1
        visiting.add(node)
        parent = mapping.get(node)
        if not parent or parent == node or parent not in mapping:
            memo[node] = 1
            visiting.discard(node)
            return 1
        d = 1 + depth(parent)
        if d > max_depth:
            d = max_depth
        memo[node] = d
        visiting.discard(node)
        return d

    depths = [depth(k) for k in mapping.keys()]
    if not depths:
        return PluginResult(
            "skipped",
            "No dependency edges to analyze",
            _basic_metrics(df, sample_meta),
            [],
            [],
            None,
        )

    max_d = int(max(depths))
    p95_d = float(np.nanpercentile(np.asarray(depths, dtype=float), 95))
    finding = {
        "id": stable_id(f"{plugin_id}:summary"),
        "kind": "dependency_critical_path",
        "severity": "warn" if max_d >= 12 else "info",
        "confidence": 0.55,
        "title": "Dependency chain depth (critical path proxy)",
        "what": f"Observed max dependency-chain depth is {max_d} steps (p95~{p95_d:.1f}).",
        "why": "Long dependency chains amplify latency and make close windows sensitive to small delays.",
        "evidence": {"metrics": {"id_col": id_col, "parent_col": parent_col, "nodes": int(len(mapping)), "max_depth": max_d, "p95_depth": p95_d}},
        "recommendations": [
            "Reduce fan-in and long sequential dependency chains where safe (parallelize or pre-stage prerequisites).",
            "Identify the deepest chains and make their prerequisites earlier or more reliable.",
        ],
        "limitations": ["Depth uses inferred ID columns; semantics may differ by ERP."],
        "measurement_type": "measured",
    }
    artifacts = [
        _artifact(ctx, plugin_id, "dependency_critical_path.json", {"nodes": int(len(mapping)), "max_depth": max_d, "p95_depth": p95_d})
    ]
    return PluginResult("ok", "Computed dependency critical path proxy", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _param_variant_explosion_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    proc_col = _pick_process_col(df, inferred)
    text_cols = [str(c) for c in (inferred.get("text_columns") or []) if str(c) in df.columns]
    cat_cols = [str(c) for c in (inferred.get("categorical_columns") or []) if str(c) in df.columns]
    param_col = _pick_col_by_name(df, text_cols + cat_cols, ("param", "descr", "args", "payload"))
    if not (proc_col and param_col):
        return PluginResult(
            "skipped",
            "Missing required columns for param variant explosion",
            _basic_metrics(df, sample_meta),
            [],
            [],
            None,
            debug={"gating_reason": "missing_columns", "proc_col": proc_col, "param_col": param_col},
        )

    proc = df[proc_col].astype(str)
    param = df[param_col].astype(str)
    ok = proc.notna() & param.notna()
    frame = pd.DataFrame({"process": proc[ok], "param": param[ok]})
    g = frame.groupby("process", dropna=False)
    summary = g.agg(
        runs=("process", "size"),
        unique_params=("param", lambda x: int(x.nunique(dropna=True))),
    ).reset_index()
    summary["unique_ratio"] = summary["unique_params"] / summary["runs"].replace(0, np.nan)
    summary = summary.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    min_runs = int(config.get("min_runs", 200))
    ratio_trigger = float(config.get("unique_ratio_trigger", 0.5))
    candidates = summary.loc[(summary["runs"] >= min_runs) & (summary["unique_ratio"] >= ratio_trigger)]
    candidates = candidates.sort_values(["unique_ratio", "runs", "process"], ascending=[False, False, True]).head(10)

    findings: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        process = str(row["process"])
        findings.append(
            {
                "id": stable_id(f"{plugin_id}:{process}"),
                "kind": "param_variant_explosion",
                "severity": "warn",
                "confidence": 0.55,
                "title": f"High parameter variety for {process}",
                "what": "This process runs with many distinct parameter variants relative to its run count.",
                "why": "High variant variety can defeat caching and create bursty, uncached workloads that inflate tail latency.",
                "evidence": {"metrics": {"process": process, "runs": int(row["runs"]), "unique_params": int(row["unique_params"]), "unique_ratio": float(row["unique_ratio"]), "param_col": param_col}},
                "recommendations": [
                    "Group and batch equivalent parameter variants where safe (reduce variant cardinality).",
                    "Add caching keyed by normalized parameters for repeated work.",
                ],
                "limitations": ["Parameter column detection is heuristic; confirm the chosen column encodes the workload key."],
                "measurement_type": "measured",
            }
        )

    artifacts = [
        _artifact(ctx, plugin_id, "param_variant_explosion.json", {"top": candidates.to_dict(orient="records"), "param_col": param_col})
    ]
    return PluginResult(
        "ok" if findings else "skipped",
        "Computed param variant explosion" if findings else "No variant explosion hotspots found",
        _basic_metrics(df, sample_meta),
        findings,
        artifacts,
        None,
    )


def _close_cycle_change_point_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    # Lightweight change detection on queue-delay/duration around close windows.
    ts_cols = [str(c) for c in (inferred.get("timestamp_columns") or []) if str(c) in df.columns]
    queue_col = _pick_col_by_name(df, ts_cols, ("queue", "queued"))
    start_col = _pick_col_by_name(df, ts_cols, ("start", "begin"))
    time_col = queue_col or start_col or _pick_col_by_name(df, ts_cols, ("time", "timestamp", "date"))
    if not time_col:
        return PluginResult(
            "skipped",
            "Missing required timestamp column for close-cycle change detection",
            _basic_metrics(df, sample_meta),
            [],
            [],
            None,
        )

    ts = coerce_datetime(df[time_col])
    ok = ts.notna()
    if ok.mean() < 0.7:
        return PluginResult(
            "skipped",
            "Low timestamp parse rate for close-cycle change detection",
            _basic_metrics(df, sample_meta),
            [],
            [],
            None,
            debug={"parse_rate": float(ok.mean()), "time_col": time_col},
        )

    metric = None
    values = None
    if queue_col and start_col:
        q = coerce_datetime(df[queue_col])
        s = coerce_datetime(df[start_col])
        ok2 = ok & q.notna() & s.notna()
        if ok2.any():
            values = (s[ok2] - q[ok2]).dt.total_seconds().clip(lower=0.0)
            metric = "queue_delay_s"
            ts_use = ts[ok2]
        else:
            ts_use = ts[ok]
    else:
        ts_use = ts[ok]

    if metric is None:
        # Fall back to a numeric duration column if present.
        num_cols = [str(c) for c in (inferred.get("numeric_columns") or []) if str(c) in df.columns]
        dur_col = _pick_col_by_name(df, num_cols, ("duration", "elapsed", "latency", "runtime", "seconds", "sec"))
        if not dur_col:
            return PluginResult("skipped", "No queue-delay/duration metric available", _basic_metrics(df, sample_meta), [], [], None)
        metric = f"{dur_col}_numeric"
        values = pd.to_numeric(df.loc[ok, dur_col], errors="coerce").dropna()
        ts_use = ts[ok].loc[values.index]

    if values is None or values.empty:
        return PluginResult("skipped", "No metric values for change detection", _basic_metrics(df, sample_meta), [], [], None)

    # Build daily p95 series.
    day = ts_use.dt.tz_convert("UTC").dt.floor("D") if ts_use.dt.tz is not None else ts_use.dt.tz_localize("UTC", nonexistent="NaT", ambiguous="NaT").dt.floor("D")
    frame = pd.DataFrame({"day": day, "v": pd.to_numeric(values, errors="coerce")})
    frame = frame.loc[frame["day"].notna() & frame["v"].notna()]
    if frame.empty:
        return PluginResult("skipped", "No usable data for change detection", _basic_metrics(df, sample_meta), [], [], None)

    series = frame.groupby("day")["v"].apply(lambda x: float(np.nanpercentile(x.to_numpy(dtype=float), 95))).sort_index()
    if series.size < 6:
        return PluginResult("skipped", "Not enough days for change detection", _basic_metrics(df, sample_meta), [], [], None)

    values_series = series.to_numpy(dtype=float)
    n = values_series.size
    k = max(1, int(n * 0.3))
    early = float(np.nanmean(values_series[:k]))
    late = float(np.nanmean(values_series[-k:]))
    ratio = late / max(early, 1e-9)
    trigger = float(config.get("drift_ratio_trigger", 1.5))
    findings = []
    if ratio >= trigger:
        findings.append(
            {
                "id": stable_id(f"{plugin_id}:drift"),
                "kind": "close_cycle_change_point",
                "severity": "warn",
                "confidence": 0.5,
                "title": "Close-cycle performance drift detected",
                "what": f"Daily p95 {metric} increased by ~{ratio:.2f}x from early to late period.",
                "why": "This suggests a structural change or load mix shift that increased close-cycle pressure.",
                "evidence": {"metrics": {"metric": metric, "early_mean_p95": early, "late_mean_p95": late, "ratio": ratio, "days": int(n), "time_col": time_col}},
                "recommendations": [
                    "Identify what changed during the drift window (deployments, data volume, schedule changes).",
                    "Use the per-process bottleneck and counterfactual plugins to isolate the driver sequence.",
                ],
                "limitations": ["This is a coarse drift heuristic; confirm with targeted changepoint methods if needed."],
                "measurement_type": "measured",
            }
        )

    artifacts = [_artifact(ctx, plugin_id, "close_cycle_change_point.json", {"metric": metric, "series": {str(k): float(v) for k, v in series.items()}, "ratio": ratio})]
    return PluginResult(
        "ok" if findings else "skipped",
        "Computed close-cycle change detection" if findings else "No close-cycle drift detected",
        _basic_metrics(df, sample_meta),
        findings,
        artifacts,
        None,
    )


HANDLERS: dict[str, Callable[..., PluginResult]] = {
    "analysis_hold_time_attribution_v1": _hold_time_attribution_v1,
    "analysis_retry_rate_hotspots_v1": _retry_rate_hotspots_v1,
    "analysis_dependency_critical_path_v1": _dependency_critical_path_v1,
    "analysis_param_variant_explosion_v1": _param_variant_explosion_v1,
    "analysis_close_cycle_change_point_v1": _close_cycle_change_point_v1,
}
