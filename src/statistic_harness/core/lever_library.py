from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import math

import numpy as np
import pandas as pd

from statistic_harness.core.ideaspace_feature_extractor import (
    IdeaspaceColumns,
    concurrency_by_bucket,
    coerce_datetime,
    duration_seconds,
    queue_delay_seconds,
)


@dataclass(frozen=True)
class LeverRecommendation:
    lever_id: str
    title: str
    action: str
    estimated_improvement_pct: float | None
    confidence: float
    evidence: dict[str, Any]
    limitations: list[str]


def _pct_improvement(current: float, ideal: float) -> float | None:
    if not (math.isfinite(current) and math.isfinite(ideal)):
        return None
    if current <= 0:
        return None
    if ideal < 0:
        return None
    improvement = (current - ideal) / current
    return float(max(0.0, min(0.95, improvement)) * 100.0)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 10 or y.size < 10:
        return None
    if np.nanstd(x) <= 0 or np.nanstd(y) <= 0:
        return None
    corr = np.corrcoef(x, y)[0, 1]
    if not math.isfinite(float(corr)):
        return None
    return float(corr)


def _maybe_peak_window(buckets: pd.DataFrame) -> tuple[pd.DataFrame, str] | tuple[None, str]:
    if buckets.empty:
        return None, "no_buckets"
    # Define "high pressure" windows as top decile active concurrency (deterministic).
    thresh = float(np.nanpercentile(buckets["active"].to_numpy(dtype=float), 90))
    peak = buckets.loc[buckets["active"] >= thresh]
    if peak.empty:
        return None, "no_peak_windows"
    return peak, ""


def recommend_workload_isolation(
    df: pd.DataFrame, cols: IdeaspaceColumns, config: dict[str, Any]
) -> LeverRecommendation | None:
    """Workload isolation via priority classes / reserved capacity."""

    min_rows = int(config.get("ideaspace_min_rows_for_reco", 200))
    if len(df) < min_rows:
        return None
    if not (cols.eligible_col and cols.start_col):
        return None
    qd = queue_delay_seconds(df, cols.eligible_col, cols.start_col)
    qd = qd[np.isfinite(qd.to_numpy(dtype=float))]
    if qd.size < min_rows:
        return None
    p95 = float(np.nanpercentile(qd.to_numpy(dtype=float), 95))
    if p95 <= float(config.get("ideaspace_queue_delay_p95_trigger_s", 120.0)):
        return None
    evidence = {"metrics": {"queue_delay_p95_s": p95, "eligible_col": cols.eligible_col, "start_col": cols.start_col}}
    return LeverRecommendation(
        lever_id="priority_isolation",
        title="Workload isolation via priority classes / reserved capacity",
        action="Introduce priority classes or reserved capacity so close-critical work is not queued behind non-critical load.",
        estimated_improvement_pct=None,
        confidence=0.6,
        evidence=evidence,
        limitations=["Does not prove causality; requires operational validation with a staged rollout."],
    )


def recommend_blackout_non_critical(
    df: pd.DataFrame, cols: IdeaspaceColumns, config: dict[str, Any]
) -> LeverRecommendation | None:
    """Close-window blackout of non-critical scheduled jobs."""

    if not (cols.start_col and cols.end_col):
        return None
    buckets = concurrency_by_bucket(
        df, cols.start_col, cols.end_col, bucket_seconds=int(config.get("ideaspace_bucket_seconds", 60))
    )
    peak, reason = _maybe_peak_window(buckets)
    if peak is None:
        return None
    # If there's a "scheduled/cron" hint column, use it; else skip.
    sched_col = None
    for col in cols.group_cols:
        if any(tok in col.lower() for tok in ("cron", "sched", "schedule", "batch", "job")):
            sched_col = col
            break
    if not sched_col or sched_col not in df.columns:
        return None
    # Proportion of scheduled work in peak windows is approximated by slicing to those buckets.
    time_col = cols.time_col or cols.start_col
    if not time_col or time_col not in df.columns:
        return None
    ts = coerce_datetime(df[time_col])
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    ts_utc = ts.dt.tz_convert("UTC") if ts.dt.tz is not None else ts.dt.tz_localize("UTC", nonexistent="NaT", ambiguous="NaT")
    bucket_idx = ((ts_utc - epoch).dt.total_seconds() // float(int(config.get("ideaspace_bucket_seconds", 60)))).astype("Int64")
    peak_set = set(int(b) for b in peak["bucket"].to_list())
    in_peak = df.loc[bucket_idx.isin(peak_set)]
    if in_peak.empty:
        return None
    ratio = float((in_peak[sched_col].astype(str).str.lower().str.contains("cron|sched|batch|job")).mean())
    if ratio < float(config.get("ideaspace_blackout_trigger_ratio", 0.25)):
        return None
    return LeverRecommendation(
        lever_id="blackout_scheduled_jobs",
        title="Close-window blackout of non-critical scheduled jobs",
        action="Black out non-critical scheduled jobs during high-pressure windows to reduce contention and queue delays.",
        estimated_improvement_pct=None,
        confidence=0.55,
        evidence={"metrics": {"peak_scheduled_ratio": ratio, "column": sched_col}},
        limitations=["Heuristic detection of scheduled jobs; may require explicit tagging."],
    )


def recommend_concurrency_cap(
    df: pd.DataFrame, cols: IdeaspaceColumns, config: dict[str, Any]
) -> LeverRecommendation | None:
    """Cap concurrency on a thrashing step."""

    if not (cols.start_col and cols.end_col):
        return None
    buckets = concurrency_by_bucket(
        df, cols.start_col, cols.end_col, bucket_seconds=int(config.get("ideaspace_bucket_seconds", 60))
    )
    if buckets.empty:
        return None
    # Throughput proxy = ended per bucket.
    x = buckets["active"].to_numpy(dtype=float)
    y = buckets["ended"].to_numpy(dtype=float)
    corr = _safe_corr(x, y)
    if corr is None or corr >= float(config.get("ideaspace_concurrency_thrash_corr_trigger", -0.15)):
        return None
    # Recommend capping at the median concurrency of buckets in the best throughput quartile.
    q = float(np.nanpercentile(y, 75))
    best = buckets.loc[buckets["ended"] >= q]
    if best.empty:
        return None
    cap = int(np.nanpercentile(best["active"].to_numpy(dtype=float), 50))
    evidence = {"metrics": {"corr_active_vs_throughput": corr, "suggested_cap": cap}}
    return LeverRecommendation(
        lever_id="cap_concurrency",
        title="Cap concurrency on a thrashing step",
        action=f"Cap concurrency near ~{cap} (approx) where throughput is highest to reduce contention-driven thrash.",
        estimated_improvement_pct=None,
        confidence=float(min(0.85, max(0.4, (-corr)))),
        evidence=evidence,
        limitations=["This uses an aggregate throughput proxy; verify per-step with tracing if available."],
    )


def recommend_split_batches(
    df: pd.DataFrame, cols: IdeaspaceColumns, config: dict[str, Any]
) -> LeverRecommendation | None:
    """Split oversized batches / transactions."""

    if not cols.batch_col:
        return None
    dur = duration_seconds(df, cols.duration_col, cols.start_col, cols.end_col)
    if dur.empty:
        return None
    b = pd.to_numeric(df[cols.batch_col], errors="coerce")
    ok = b.notna() & dur.notna()
    if ok.mean() < 0.5:
        return None
    b = b[ok].to_numpy(dtype=float)
    d = dur[ok].to_numpy(dtype=float)
    if b.size < int(config.get("ideaspace_min_rows_for_reco", 200)):
        return None
    # Tail amplification heuristic: compare p95 duration in top decile batches vs median batches.
    thresh = float(np.nanpercentile(b, 90))
    top = d[b >= thresh]
    mid = d[b <= float(np.nanpercentile(b, 50))]
    if top.size < 20 or mid.size < 20:
        return None
    p95_top = float(np.nanpercentile(top, 95))
    p95_mid = float(np.nanpercentile(mid, 95))
    if p95_top <= p95_mid * float(config.get("ideaspace_batch_tail_multiplier_trigger", 1.5)):
        return None
    evidence = {"metrics": {"batch_col": cols.batch_col, "p95_top": p95_top, "p95_mid": p95_mid}}
    return LeverRecommendation(
        lever_id="split_batches",
        title="Split oversized batches / transactions",
        action="Split oversized batches to reduce tail latency amplification.",
        estimated_improvement_pct=float(max(0.0, min(0.9, (p95_top - p95_mid) / max(p95_top, 1e-9))) * 100.0),
        confidence=0.6,
        evidence=evidence,
        limitations=["Batch column detection is heuristic; confirm semantics of the batch size field."],
    )


def recommend_retry_backoff(
    df: pd.DataFrame, cols: IdeaspaceColumns, config: dict[str, Any]
) -> LeverRecommendation | None:
    """Introduce retry backoff / circuit breaker for hot failure templates."""

    if not (cols.text_col and cols.time_col):
        return None
    min_rows = int(config.get("ideaspace_min_rows_for_reco", 200))
    if len(df) < min_rows:
        return None
    ts = coerce_datetime(df[cols.time_col])
    msg = df[cols.text_col].astype(str)
    ok = ts.notna() & msg.notna()
    if ok.mean() < 0.6:
        return None
    ts = ts[ok]
    msg = msg[ok].str.lower()
    # Hot-template burst heuristic: top template frequency within short buckets is high.
    bucket_s = int(config.get("ideaspace_retry_bucket_seconds", 60))
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    ts_utc = ts.dt.tz_convert("UTC") if ts.dt.tz is not None else ts.dt.tz_localize("UTC", nonexistent="NaT", ambiguous="NaT")
    bucket = ((ts_utc - epoch).dt.total_seconds() // float(bucket_s)).astype(int)
    frame = pd.DataFrame({"bucket": bucket, "msg": msg})
    counts = frame.groupby(["bucket", "msg"]).size().reset_index(name="count")
    if counts.empty:
        return None
    worst = counts.sort_values(["count", "msg"], ascending=[False, True]).head(1)
    worst_count = int(worst["count"].iloc[0])
    if worst_count < int(config.get("ideaspace_retry_burst_count_trigger", 10)):
        return None
    evidence = {"metrics": {"text_col": cols.text_col, "bucket_seconds": bucket_s, "worst_bucket_count": worst_count}}
    return LeverRecommendation(
        lever_id="retry_backoff",
        title="Introduce retry backoff / circuit breaker for hot failure templates",
        action="Add deterministic retry backoff/circuit breaker around the hottest repeated failure templates to reduce self-inflicted load.",
        estimated_improvement_pct=None,
        confidence=0.55,
        evidence=evidence,
        limitations=["This is based on message repetition; confirm retries vs independent failures."],
    )


def recommend_resource_affinity(
    df: pd.DataFrame, cols: IdeaspaceColumns, config: dict[str, Any]
) -> LeverRecommendation | None:
    """Resource affinity / pinning to reduce cold-start penalties."""

    if not (cols.host_col and cols.duration_col):
        return None
    dur = pd.to_numeric(df[cols.duration_col], errors="coerce")
    host = df[cols.host_col].astype(str)
    ok = dur.notna() & host.notna()
    if ok.mean() < 0.6:
        return None
    dur = dur[ok].to_numpy(dtype=float)
    host = host[ok]
    if host.nunique() < 2:
        return None
    # Host effect heuristic: compare p50 between best and median host.
    frame = pd.DataFrame({"host": host, "dur": dur})
    med_by_host = frame.groupby("host")["dur"].median().sort_values()
    if len(med_by_host) < 2:
        return None
    best = float(med_by_host.iloc[0])
    med = float(med_by_host.iloc[len(med_by_host) // 2])
    if med <= 0:
        return None
    if best >= med * float(config.get("ideaspace_affinity_multiplier_trigger", 0.9)):
        return None
    evidence = {"metrics": {"duration_col": cols.duration_col, "host_col": cols.host_col, "best_host_median": best, "median_host_median": med}}
    return LeverRecommendation(
        lever_id="resource_affinity",
        title="Resource affinity / pinning to reduce cold-start penalties",
        action="Use resource affinity/pinning (where safe) so repeated work lands on hosts with materially lower service times.",
        estimated_improvement_pct=_pct_improvement(med, best),
        confidence=0.5,
        evidence=evidence,
        limitations=["Host effect could be confounded by workload mix; validate with controlled routing."],
    )


def recommend_parallelize_branches(
    df: pd.DataFrame, cols: IdeaspaceColumns, config: dict[str, Any]
) -> LeverRecommendation | None:
    """Parallelize independent branches (dependency/sequence)."""

    if not (cols.case_col and cols.activity_col and cols.time_col):
        return None
    ts = coerce_datetime(df[cols.time_col])
    ok = ts.notna()
    if ok.mean() < 0.7:
        return None
    frame = df.loc[ok, [cols.case_col, cols.activity_col]].copy()
    frame["_t"] = ts.loc[ok]
    # For each case, compute observed precedence pairs.
    pairs: dict[tuple[str, str], int] = {}
    for _, g in frame.sort_values("_t").groupby(cols.case_col):
        seq = g[cols.activity_col].astype(str).tolist()
        for a, b in zip(seq, seq[1:], strict=False):
            pairs[(a, b)] = pairs.get((a, b), 0) + 1
    if not pairs:
        return None
    # Identify "weakly ordered" activities: both A->B and B->A appear.
    for (a, b), ab in sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0])):
        ba = pairs.get((b, a), 0)
        if ba <= 0:
            continue
        total = ab + ba
        if total < int(config.get("ideaspace_parallelize_pair_min", 25)):
            continue
        balance = min(ab, ba) / max(total, 1)
        if balance < float(config.get("ideaspace_parallelize_balance_trigger", 0.25)):
            continue
        evidence = {"metrics": {"case_col": cols.case_col, "activity_col": cols.activity_col, "pair": [a, b], "ab": ab, "ba": ba}}
        return LeverRecommendation(
            lever_id="parallelize_branches",
            title="Parallelize independent branches",
            action="Where safe, parallelize weakly-ordered prerequisite steps so total makespan approaches max(branch) rather than sum(branch).",
            estimated_improvement_pct=None,
            confidence=0.45,
            evidence=evidence,
            limitations=["Weak ordering does not prove independence; validate resource constraints and correctness dependencies."],
        )
    return None


def recommend_prestage_prereqs(
    df: pd.DataFrame, cols: IdeaspaceColumns, config: dict[str, Any]
) -> LeverRecommendation | None:
    """Pre-stage invariant close prerequisites earlier."""

    if not (cols.time_col and cols.activity_col):
        return None
    ts = coerce_datetime(df[cols.time_col])
    ok = ts.notna()
    if ok.mean() < 0.7:
        return None
    bucket_s = int(config.get("ideaspace_bucket_seconds", 60))
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    ts_utc = ts.loc[ok].dt.tz_convert("UTC") if ts.loc[ok].dt.tz is not None else ts.loc[ok].dt.tz_localize("UTC", nonexistent="NaT", ambiguous="NaT")
    bucket = ((ts_utc - epoch).dt.total_seconds() // float(bucket_s)).astype(int)
    frame = pd.DataFrame({"bucket": bucket, "activity": df.loc[ok, cols.activity_col].astype(str)})
    counts = frame.groupby(["bucket", "activity"]).size().reset_index(name="count")
    if counts.empty:
        return None
    # Peak is top decile bucket count overall.
    total = frame.groupby("bucket").size()
    thresh = float(np.nanpercentile(total.to_numpy(dtype=float), 90))
    peak_buckets = set(int(b) for b, c in total.items() if float(c) >= thresh)
    if not peak_buckets:
        return None
    in_peak = counts.loc[counts["bucket"].isin(peak_buckets)]
    if in_peak.empty:
        return None
    top = in_peak.sort_values(["count", "activity"], ascending=[False, True]).head(1)
    if top.empty:
        return None
    activity = str(top["activity"].iloc[0])
    c = int(top["count"].iloc[0])
    if c < int(config.get("ideaspace_prestage_min_count", 20)):
        return None
    return LeverRecommendation(
        lever_id="prestage_prereqs",
        title="Pre-stage invariant prerequisites earlier",
        action="Pre-stage invariant prerequisites earlier to smooth the start-of-window surge and reduce peak contention.",
        estimated_improvement_pct=None,
        confidence=0.45,
        evidence={"metrics": {"activity": activity, "peak_count": c, "bucket_seconds": bucket_s}},
        limitations=["Requires identifying tasks that are safe to shift earlier (no data dependencies)."],
    )


def build_default_lever_recommendations(
    df: pd.DataFrame,
    cols: IdeaspaceColumns,
    config: dict[str, Any],
) -> list[LeverRecommendation]:
    recos: list[LeverRecommendation] = []
    for fn in (
        recommend_workload_isolation,
        recommend_blackout_non_critical,
        recommend_concurrency_cap,
        recommend_split_batches,
        recommend_retry_backoff,
        recommend_resource_affinity,
        recommend_parallelize_branches,
        recommend_prestage_prereqs,
    ):
        try:
            reco = fn(df, cols, config)
        except Exception:  # pragma: no cover - defense in depth
            reco = None
        if reco is not None:
            recos.append(reco)
    return recos
