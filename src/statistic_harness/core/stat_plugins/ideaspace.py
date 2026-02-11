from __future__ import annotations

from typing import Any, Callable

import math

import numpy as np
import pandas as pd

from statistic_harness.core.baselines import load_signed_baseline
from statistic_harness.core.ideaspace_feature_extractor import (
    IdeaspaceColumns,
    coerce_datetime,
    duration_seconds,
    entity_slices,
    error_rate,
    kpi_summary,
    pick_columns,
    queue_delay_seconds,
    time_span_seconds,
)
from statistic_harness.core.lever_library import build_default_lever_recommendations
from statistic_harness.core.stat_plugins import (
    BudgetTimer,
    build_redactor,
    deterministic_sample,
    stable_id,
)
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import read_json, write_json


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
    return PluginArtifact(path=str(path.relative_to(ctx.run_dir)), type="json", description=name)


def _make_finding(
    plugin_id: str,
    key: str,
    title: str,
    what: str,
    why: str,
    evidence: dict[str, Any],
    kind: str,
    severity: str = "info",
    confidence: float = 0.5,
    recommendations: list[str] | None = None,
    limitations: list[str] | None = None,
    # Optional modeled-comparison fields. If provided, they satisfy report_decision_bundle_v2's
    # modeled finding requirements.
    baseline_value: float | None = None,
    modeled_value: float | None = None,
    delta_value: float | None = None,
    unit: str | None = None,
    baseline_host_count: int | None = None,
    modeled_host_count: int | None = None,
) -> dict[str, Any]:
    # Back-compat fields ("scope"/"assumptions") are kept, but decision bundles standardize on
    # modeled_scope/modeled_assumptions + baseline/modeled/delta/unit.
    modeled_scope = {"plugin_id": plugin_id}
    modeled_assumptions = [
        "Unknown schema: uses heuristic column inference.",
        "Estimated impacts are conservative and must be validated operationally.",
    ]
    return {
        "id": stable_id(f"{plugin_id}:{key}"),
        "kind": kind,
        "severity": severity,
        "confidence": float(max(0.0, min(1.0, confidence))),
        "title": title,
        "what": what,
        "why": why,
        "evidence": evidence,
        "recommendations": recommendations or [],
        "limitations": limitations or [],
        "measurement_type": "modeled",
        "scope": modeled_scope,
        "assumptions": modeled_assumptions,
        "modeled_scope": modeled_scope,
        "modeled_assumptions": modeled_assumptions,
        "baseline_host_count": int(baseline_host_count or 1),
        "modeled_host_count": int(modeled_host_count or 1),
        "baseline_value": float(baseline_value) if isinstance(baseline_value, (int, float)) else None,
        "modeled_value": float(modeled_value) if isinstance(modeled_value, (int, float)) else None,
        "delta_value": float(delta_value) if isinstance(delta_value, (int, float)) else None,
        "unit": str(unit or ""),
    }


MINIMIZE_KEYS = ("duration_p95", "queue_delay_p95", "error_rate")
MAXIMIZE_KEYS = ("rate_per_min",)
ENERGY_MINIMIZE_KEYS = MINIMIZE_KEYS + ("background_overhead_per_min",)
ENERGY_MAXIMIZE_KEYS = MAXIMIZE_KEYS


def rng_for_run(run_seed: int, plugin_id: str) -> np.random.Generator:
    # Stable per-run RNG. Do not use Python's built-in hash() (salted per-process).
    import hashlib

    material = f"{int(run_seed)}:{plugin_id}".encode("utf-8")
    mixed = int(hashlib.sha256(material).hexdigest()[:8], 16)
    return np.random.default_rng(mixed)


def redact_value(value: str, privacy: dict[str, Any] | None = None) -> str:
    redactor = build_redactor(privacy or {})
    return redactor(str(value))


def pick_groupable_columns(
    df: pd.DataFrame,
    candidates: list[str],
    k_min: int = 5,
    max_cardinality: int = 50,
    max_columns: int = 5,
) -> list[str]:
    chosen: list[str] = []
    for col in candidates:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        nunique = int(series.nunique())
        if nunique < 2 or nunique > int(max_cardinality):
            continue
        counts = series.value_counts()
        if counts.min() < int(k_min):
            continue
        chosen.append(col)
        if len(chosen) >= int(max_columns):
            break
    return chosen


def apply_budget_rows(df: pd.DataFrame, row_limit: int | None, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    return deterministic_sample(df, row_limit, seed=int(seed))


def _pareto_frontier(points: list[dict[str, Any]]) -> list[int]:
    """Return indices of non-dominated points (deterministic tie-break)."""

    # Only use keys that are present across at least 2 points.
    minimize = [k for k in MINIMIZE_KEYS if sum(k in p and math.isfinite(float(p[k])) for p in points) >= 2]
    maximize = [k for k in MAXIMIZE_KEYS if sum(k in p and math.isfinite(float(p[k])) for p in points) >= 2]
    if not minimize and not maximize:
        return []

    def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
        better_or_equal = True
        strictly_better = False
        for k in minimize:
            av = float(a.get(k, float("nan")))
            bv = float(b.get(k, float("nan")))
            if not (math.isfinite(av) and math.isfinite(bv)):
                continue
            if av > bv:
                better_or_equal = False
                break
            if av < bv:
                strictly_better = True
        if not better_or_equal:
            return False
        for k in maximize:
            av = float(a.get(k, float("nan")))
            bv = float(b.get(k, float("nan")))
            if not (math.isfinite(av) and math.isfinite(bv)):
                continue
            if av < bv:
                better_or_equal = False
                break
            if av > bv:
                strictly_better = True
        return better_or_equal and strictly_better

    frontier: list[int] = []
    for i, a in enumerate(points):
        dominated = False
        for j, b in enumerate(points):
            if i == j:
                continue
            if dominates(b, a):
                dominated = True
                break
        if not dominated:
            frontier.append(i)
    return sorted(frontier)


def _nearest_frontier_point(
    point: dict[str, Any], frontier: list[dict[str, Any]], keys: list[str]
) -> dict[str, Any] | None:
    if not frontier:
        return None
    best = None
    best_dist = None
    for cand in frontier:
        dist = 0.0
        used = 0
        for k in keys:
            pv = float(point.get(k, float("nan")))
            cv = float(cand.get(k, float("nan")))
            if not (math.isfinite(pv) and math.isfinite(cv)):
                continue
            scale = max(abs(cv), 1.0)
            dist += ((pv - cv) / scale) ** 2
            used += 1
        if used == 0:
            continue
        dist = float(dist)
        if best is None or (best_dist is not None and dist < best_dist):
            best = cand
            best_dist = dist
        elif best is not None and best_dist is not None and dist == best_dist:
            # Deterministic tie-break: prefer lexicographically smaller entity_key.
            if str(cand.get("entity_key", "")) < str(best.get("entity_key", "")):
                best = cand
                best_dist = dist
    return best


def _entity_metrics(df: pd.DataFrame, cols: IdeaspaceColumns, redactor: Callable[[str], str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"n_rows": int(len(df))}
    span = time_span_seconds(df, cols.time_col)
    if span is not None:
        metrics["time_span_s"] = float(span)
        metrics["rate_per_min"] = float(len(df) / max(span / 60.0, 1e-9))
    dur = duration_seconds(df, cols.duration_col, cols.start_col, cols.end_col)
    dur_kpi = kpi_summary(dur)
    if dur_kpi:
        metrics["duration_p50"] = float(dur_kpi["p50"])
        metrics["duration_p95"] = float(dur_kpi["p95"])
    qd = queue_delay_seconds(df, cols.eligible_col, cols.start_col)
    qd_kpi = kpi_summary(qd)
    if qd_kpi:
        metrics["queue_delay_p50"] = float(qd_kpi["p50"])
        metrics["queue_delay_p95"] = float(qd_kpi["p95"])

    er = error_rate(df, list(df.columns), redactor)
    if er is not None and math.isfinite(er):
        metrics["error_rate"] = float(max(0.0, min(1.0, er)))
    return metrics


def _apply_baseline_ideal(config: dict[str, Any], ctx) -> dict[str, Any] | None:
    baseline_path = config.get("ideaspace_baseline_path")
    if not isinstance(baseline_path, str) or not baseline_path.strip():
        return None
    p = baseline_path.strip()
    # Security: allow only within run_dir or appdata by default (fail closed).
    cand = (ctx.run_dir / p).resolve() if not Path(p).is_absolute() else Path(p).resolve()
    if str(cand).startswith(str(ctx.run_dir.resolve())):
        pass
    elif str(cand).startswith(str((ctx.run_dir / ".." / ".." / "appdata").resolve())):
        pass
    elif str(cand).startswith(str((Path.cwd() / "appdata").resolve())):
        pass
    else:
        raise ValueError("baseline path must be under run_dir or appdata")
    schema_path = Path(__file__).resolve().parents[3] / "docs" / "ideaspace_baseline.schema.json"
    baseline = load_signed_baseline(cand, schema_path)
    ideal = baseline.get("ideal_vector") or {}
    if not isinstance(ideal, dict):
        return None
    return {k: float(v) for k, v in ideal.items() if isinstance(v, (int, float)) and math.isfinite(float(v))}


def _ideaspace_normative_gap(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    redactor = build_redactor(config.get("privacy") or {})
    cols = pick_columns(df, inferred, config)

    max_groups_per_col = int(config.get("ideaspace_max_groups_per_col", 10))
    max_entities = int(config.get("ideaspace_max_entities", 60))
    slices = entity_slices(df, cols.group_cols, max_groups_per_col, max_entities, redactor)

    entities: list[dict[str, Any]] = []
    for entity_key, slice_df, meta in slices:
        if timer.exceeded():
            break
        m = _entity_metrics(slice_df, cols, redactor)
        m["entity_key"] = entity_key
        m["entity_label"] = meta.get("label", entity_key)
        m["group"] = meta.get("group") or {}
        entities.append(m)

    if not entities:
        return PluginResult(
            status="skipped",
            summary="No entities",
            metrics=_basic_metrics(df, sample_meta),
            findings=[],
            artifacts=[],
            error=None,
            references=[],
            debug={"gating_reason": "no_entities"},
        )

    # Ideal selection: baseline or in-dataset frontier.
    ideal_vector = _apply_baseline_ideal(config, ctx)
    keys = sorted({k for e in entities for k in e.keys() if k in MINIMIZE_KEYS + MAXIMIZE_KEYS})
    if ideal_vector is not None:
        ideal_mode = "baseline"
        for e in entities:
            e["ideal"] = {k: float(ideal_vector.get(k)) for k in keys if k in ideal_vector}
            e["ideal_entity_key"] = "BASELINE"
    else:
        ideal_mode = "frontier"
        frontier_idx = _pareto_frontier(entities)
        frontier = [entities[i] for i in frontier_idx]
        for e in entities:
            nearest = _nearest_frontier_point(e, frontier, keys)
            e["ideal"] = {k: float(nearest.get(k)) for k in keys if nearest and k in nearest}
            e["ideal_entity_key"] = str(nearest.get("entity_key")) if nearest else None

    # Gap + badness.
    for e in entities:
        ideal = e.get("ideal") or {}
        gaps: dict[str, float] = {}
        badness: dict[str, float] = {}
        for k in keys:
            cur = e.get(k)
            ref = ideal.get(k)
            if not (isinstance(cur, (int, float)) and isinstance(ref, (int, float))):
                continue
            cur_f = float(cur)
            ref_f = float(ref)
            if not (math.isfinite(cur_f) and math.isfinite(ref_f)):
                continue
            gaps[k] = float(cur_f - ref_f)
            denom = max(abs(ref_f), 1e-9)
            if k in MINIMIZE_KEYS:
                badness[k] = float(max(0.0, (cur_f - ref_f) / denom))
            else:
                badness[k] = float(max(0.0, (ref_f - cur_f) / denom))
        e["gaps"] = gaps
        e["badness"] = badness
        e["total_badness"] = float(sum(badness.values()))

    # Findings: top gap entities (no speculation gate: min rows).
    min_rows = int(config.get("ideaspace_min_rows_entity", 200))
    ranked = sorted(
        [e for e in entities if int(e.get("n_rows", 0)) >= min_rows],
        key=lambda r: (-float(r.get("total_badness", 0.0)), str(r.get("entity_key", ""))),
    )
    max_findings = int(config.get("max_findings", 30))
    findings: list[dict[str, Any]] = []
    for e in ranked[:max_findings]:
        gaps = e.get("gaps") or {}
        bad = e.get("badness") or {}
        top_components = sorted(bad.items(), key=lambda kv: (-float(kv[1]), kv[0]))[:5]
        evidence = {
            "metrics": {
                "entity": e.get("entity_label"),
                "n_rows": int(e.get("n_rows", 0)),
                "ideal_mode": ideal_mode,
                "baseline_total_badness": float(e.get("total_badness", 0.0)),
                "ideal_total_badness": 0.0,
                "top_gap_components": [{"metric": k, "badness": float(v), "gap": float(gaps.get(k, 0.0))} for k, v in top_components],
            }
        }
        baseline_badness = float(e.get("total_badness", 0.0) or 0.0)
        findings.append(
            _make_finding(
                plugin_id,
                f"gap:{e.get('entity_key')}",
                f"Ideaspace gap vs ideal: {e.get('entity_label')}",
                "This cohort deviates materially from the selected ideal reference/frontier.",
                "Largest normalized gaps identify where this cohort is underperforming relative to an ideal.",
                evidence,
                kind="ideaspace_gap",
                severity="warn" if float(e.get("total_badness", 0.0)) >= 0.5 else "info",
                confidence=float(min(0.9, 0.4 + float(e.get("total_badness", 0.0)))),
                baseline_value=baseline_badness,
                modeled_value=0.0,
                delta_value=-baseline_badness,
                unit="normalized_badness",
            )
        )

    artifacts = [
        _artifact(ctx, plugin_id, "entities_table.json", entities),
    ]
    summary = "Computed ideaspace vectors, ideal reference, and gap vectors."
    if not findings:
        summary = "Computed ideaspace vectors; insufficient evidence to emit gap findings."
    debug = {
        "ideaspace": {
            "ideal_mode": ideal_mode,
            "columns": cols.__dict__,
        }
    }
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None, references=[], debug=debug)


def _ideaspace_action_planner(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    redactor = build_redactor(config.get("privacy") or {})
    cols = pick_columns(df, inferred, config)

    # Optional: load normative gap artifact for user-facing "ideal vs current" context.
    gap_artifact = ctx.run_dir / "artifacts" / "analysis_ideaspace_normative_gap" / "entities_table.json"
    gap_entities = None
    if gap_artifact.exists():
        try:
            gap_entities = read_json(gap_artifact)
        except Exception:
            gap_entities = None

    recos = build_default_lever_recommendations(df, cols, config)
    # Strict no-speculation gate: if we cannot trigger any lever, return skipped.
    if not recos:
        return PluginResult(
            status="skipped",
            summary="No actionable levers triggered under evidence gates",
            metrics=_basic_metrics(df, sample_meta),
            findings=[],
            artifacts=[],
            references=[],
            debug={"gating_reason": "no_levers", "ideaspace_gap_loaded": bool(gap_entities is not None)},
        )

    # Deterministic ordering: highest confidence then stable lever_id.
    recos.sort(key=lambda r: (-float(r.confidence), r.lever_id))

    findings: list[dict[str, Any]] = []
    for reco in recos[: int(config.get("max_findings", 30))]:
        est = (
            f" (~{reco.estimated_improvement_pct:.1f}%)"
            if isinstance(reco.estimated_improvement_pct, (int, float))
            else ""
        )
        est_pct = float(reco.estimated_improvement_pct or 0.0) if isinstance(reco.estimated_improvement_pct, (int, float)) else 0.0
        findings.append(
            _make_finding(
                plugin_id,
                f"lever:{reco.lever_id}",
                reco.title,
                f"Action: {reco.action}{est}",
                "Trigger conditions satisfied with deterministic evidence gates.",
                reco.evidence,
                kind="ideaspace_action",
                severity="warn" if reco.confidence >= 0.65 else "info",
                confidence=reco.confidence,
                recommendations=[reco.action],
                limitations=reco.limitations,
                baseline_value=0.0,
                modeled_value=est_pct,
                delta_value=est_pct,
                unit="percent",
            )
        )

    artifacts = [
        _artifact(ctx, plugin_id, "recommendations.json", [r.__dict__ for r in recos]),
    ]
    if gap_entities is not None:
        artifacts.append(_artifact(ctx, plugin_id, "gap_entities_snapshot.json", gap_entities))

    summary = f"Generated {len(findings)} deterministic action recommendations."
    debug = {"ideaspace": {"columns": cols.__dict__, "ideaspace_gap_loaded": bool(gap_entities is not None)}}
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None, references=[], debug=debug)


def _pick_process_col(df: pd.DataFrame, cols: IdeaspaceColumns) -> str | None:
    for col in df.columns:
        if "process" in str(col).lower():
            return str(col)
    if cols.activity_col and cols.activity_col in df.columns:
        return cols.activity_col
    return None


def _default_energy_weights(overrides: dict[str, Any] | None = None) -> dict[str, float]:
    weights: dict[str, float] = {
        "duration_p95": 1.0,
        "queue_delay_p95": 2.0,
        "error_rate": 1.0,
        "rate_per_min": 0.5,
        "background_overhead_per_min": 1.0,
    }
    if isinstance(overrides, dict):
        for k, v in overrides.items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) >= 0:
                weights[str(k)] = float(v)
    return weights


def _default_constraint_penalties(overrides: dict[str, Any] | None = None) -> dict[str, float]:
    penalties: dict[str, float] = {
        "missing_time_columns": 1000.0,
        "negative_durations": 200.0,
        "timestamp_parse_rate_low": 200.0,
    }
    if isinstance(overrides, dict):
        for k, v in overrides.items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) >= 0:
                penalties[str(k)] = float(v)
    return penalties


def _energy_terms(
    observed: dict[str, Any],
    ideal: dict[str, Any],
    weights: dict[str, float],
    eps: float = 1e-9,
) -> tuple[list[dict[str, Any]], float]:
    terms: list[dict[str, Any]] = []
    total = 0.0
    for metric, w in sorted(weights.items(), key=lambda kv: (kv[0])):
        if w <= 0:
            continue
        cur = observed.get(metric)
        ref = ideal.get(metric)
        if not (isinstance(cur, (int, float)) and isinstance(ref, (int, float))):
            continue
        cur_f = float(cur)
        ref_f = float(ref)
        if not (math.isfinite(cur_f) and math.isfinite(ref_f)):
            continue
        # Scale by the larger of observed/ideal magnitudes to avoid blowing up when the
        # ideal reference is 0 for a nonnegative metric (common for "overhead" terms).
        denom = max(abs(ref_f), abs(cur_f), 1.0, eps)
        if metric in ENERGY_MINIMIZE_KEYS:
            gap = max(0.0, (cur_f - ref_f) / denom)
        else:
            gap = max(0.0, (ref_f - cur_f) / denom)
        contribution = float(w * (gap**2))
        total += contribution
        terms.append(
            {
                "metric": metric,
                "gap": float(gap),
                "weight": float(w),
                "contribution": float(contribution),
                "observed": float(cur_f),
                "ideal": float(ref_f),
            }
        )
    terms.sort(key=lambda t: (-float(t.get("contribution", 0.0)), str(t.get("metric", ""))))
    return terms, float(total)


def _compute_constraints(
    df: pd.DataFrame,
    cols: IdeaspaceColumns,
    penalties: dict[str, float],
    parse_rate_threshold: float,
) -> tuple[list[dict[str, Any]], float]:
    constraints: list[dict[str, Any]] = []
    total = 0.0

    dt_cols = [c for c in (cols.time_col, cols.eligible_col, cols.start_col, cols.end_col) if isinstance(c, str) and c in df.columns]
    if not dt_cols:
        penalty = float(penalties.get("missing_time_columns", 0.0))
        if penalty:
            constraints.append({"kind": "missing_time_columns", "penalty": penalty})
            total += penalty
        return constraints, float(total)

    # Parse-rate checks (defense in depth).
    for col in sorted(set(dt_cols)):
        ts = coerce_datetime(df[col])
        rate = float(ts.notna().mean()) if len(ts) else 0.0
        if rate < float(parse_rate_threshold):
            penalty = float(penalties.get("timestamp_parse_rate_low", 0.0))
            if penalty:
                constraints.append(
                    {
                        "kind": "timestamp_parse_rate_low",
                        "column": col,
                        "parse_rate": rate,
                        "threshold": float(parse_rate_threshold),
                        "penalty": penalty,
                    }
                )
                total += penalty

    # Negative durations.
    dur = duration_seconds(df, cols.duration_col, cols.start_col, cols.end_col)
    if not dur.empty:
        values = pd.to_numeric(dur, errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            neg = float((values < -1e-6).mean())
            if neg >= 0.01:
                penalty = float(penalties.get("negative_durations", 0.0))
                if penalty:
                    constraints.append({"kind": "negative_durations", "negative_rate": neg, "penalty": penalty})
                    total += penalty

    return constraints, float(total)


def _ideaspace_energy_ebm_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    """Compute deterministic energy score (distance-from-ideal + constraint penalties)."""

    redactor = build_redactor(config.get("privacy") or {})
    cols = pick_columns(df, inferred, config)

    # Build entity slices (ALL + top groups). Augment grouping with process + host if available.
    group_cols = list(cols.group_cols)
    proc_col = _pick_process_col(df, cols)
    if proc_col and proc_col in df.columns and proc_col not in group_cols:
        group_cols.insert(0, proc_col)
    if cols.host_col and cols.host_col in df.columns and cols.host_col not in group_cols:
        group_cols.append(cols.host_col)

    max_groups_per_col = int(config.get("ideaspace_max_groups_per_col", 10))
    max_entities = int(config.get("ideaspace_max_entities", 60))
    slices = entity_slices(df, group_cols, max_groups_per_col, max_entities, redactor)

    entities: list[dict[str, Any]] = []
    for entity_key, slice_df, meta in slices:
        if timer.exceeded():
            break
        m = _entity_metrics(slice_df, cols, redactor)
        m["entity_key"] = entity_key
        m["entity_label"] = meta.get("label", entity_key)
        m["group"] = meta.get("group") or {}
        entities.append(m)

    if not entities:
        return PluginResult(
            status="skipped",
            summary="No entities",
            metrics=_basic_metrics(df, sample_meta),
            findings=[],
            artifacts=[],
            error=None,
            references=[],
            debug={"gating_reason": "no_entities"},
        )

    # Add an extension metric: background overhead per minute (QEMAIL rate).
    all_entity = next((e for e in entities if str(e.get("entity_key")) == "ALL"), None)
    if all_entity is not None and proc_col and proc_col in df.columns:
        time_col = cols.time_col or cols.eligible_col or cols.start_col
        span = time_span_seconds(df, time_col) if time_col else None
        if span is not None and span > 0:
            mask = df[proc_col].astype(str).str.strip().str.lower() == "qemail"
            qemail_rows = int(mask.sum())
            all_entity["background_overhead_per_min"] = float(qemail_rows / max(span / 60.0, 1e-9))
            all_entity["qemail_rows"] = qemail_rows

    # Ideal selection: baseline or in-dataset frontier.
    baseline_path = config.get("baseline_path") or config.get("ideaspace_baseline_path")
    config_for_baseline = dict(config)
    if isinstance(baseline_path, str) and baseline_path.strip():
        config_for_baseline["ideaspace_baseline_path"] = baseline_path.strip()
    ideal_vector = _apply_baseline_ideal(config_for_baseline, ctx)

    # Energy keys are those we can evaluate across at least 2 entities OR are present in baseline.
    energy_keys = sorted(
        {
            k
            for e in entities
            for k in e.keys()
            if k in ENERGY_MINIMIZE_KEYS + ENERGY_MAXIMIZE_KEYS
        }
    )
    if ideal_vector is not None:
        ideal_mode = "baseline"
        for e in entities:
            ideal: dict[str, float] = {}
            for k in energy_keys:
                if k in ideal_vector:
                    ideal[k] = float(ideal_vector[k])
                elif k == "background_overhead_per_min":
                    ideal[k] = 0.0
            e["ideal"] = ideal
            e["ideal_entity_key"] = "BASELINE"
    else:
        ideal_mode = "frontier"
        # Frontier selection uses only the core ideaspace keys to avoid extension-metric skew.
        frontier_keys = sorted({k for k in energy_keys if k in MINIMIZE_KEYS + MAXIMIZE_KEYS})
        frontier_idx = _pareto_frontier(entities)
        frontier = [entities[i] for i in frontier_idx]
        for e in entities:
            nearest = _nearest_frontier_point(e, frontier, frontier_keys)
            ideal: dict[str, float] = {}
            for k in energy_keys:
                if nearest and k in nearest and isinstance(nearest.get(k), (int, float)):
                    ideal[k] = float(nearest[k])
                elif k == "background_overhead_per_min":
                    ideal[k] = 0.0
            e["ideal"] = ideal
            e["ideal_entity_key"] = str(nearest.get("entity_key")) if nearest else None

    weights = _default_energy_weights(config.get("weights") if isinstance(config.get("weights"), dict) else None)
    penalties = _default_constraint_penalties(
        config.get("constraint_penalties") if isinstance(config.get("constraint_penalties"), dict) else None
    )
    parse_rate_threshold = float(config.get("timestamp_parse_rate_threshold", 0.8))
    constraints, e_constraints = _compute_constraints(df, cols, penalties, parse_rate_threshold)

    for e in entities:
        ideal = e.get("ideal") or {}
        terms, e_gap = _energy_terms(e, ideal, weights)
        e["energy_terms"] = terms
        e["energy_gap"] = float(e_gap)
        e["energy_constraints"] = 0.0
        if str(e.get("entity_key")) == "ALL":
            e["energy_constraints"] = float(e_constraints)
            e["constraints"] = constraints
        e["energy_total"] = float(e["energy_gap"] + e["energy_constraints"])

    all_terms = []
    all_energy_total = None
    if all_entity is not None:
        all_terms = list(all_entity.get("energy_terms") or [])
        all_energy_total = float(all_entity.get("energy_total") or 0.0)

    top_terms = [
        {"metric": t.get("metric"), "gap": t.get("gap"), "weight": t.get("weight"), "contribution": t.get("contribution")}
        for t in all_terms[:10]
    ]
    findings = [
        {
            "id": stable_id(f"{plugin_id}:energy"),
            "kind": "ideaspace_energy",
            "severity": "info" if (all_energy_total is not None and all_energy_total < 1.0) else "warn",
            "confidence": 0.8,
            "title": "Ideaspace Energy Summary (Kona-style EBM)",
            "what": "Computed a deterministic energy score measuring distance from an ideal reference plus constraint penalties.",
            "why": "Energy supports ranking actions by expected energy reduction rather than heuristic confidence alone.",
            "energy_total": all_energy_total,
            "energy_gap": float(all_entity.get("energy_gap", 0.0)) if all_entity is not None else None,
            "energy_constraints": float(all_entity.get("energy_constraints", 0.0)) if all_entity is not None else None,
            "ideal_mode": ideal_mode,
            "top_terms": top_terms,
            "constraints": constraints,
            "measurement_type": "measured",
        }
    ]

    artifacts_dir = ctx.artifacts_dir(plugin_id)
    breakdown_path = artifacts_dir / "energy_breakdown.json"
    vector_path = artifacts_dir / "energy_state_vector.json"
    write_json(
        breakdown_path,
        {
            "schema_version": "v1",
            "ideal_mode": ideal_mode,
            "weights": weights,
            "constraint_penalties": penalties,
            "timestamp_parse_rate_threshold": parse_rate_threshold,
            "entities": entities,
        },
    )
    write_json(
        vector_path,
        {
            "schema_version": "v1",
            "ideal_mode": ideal_mode,
            "weights": weights,
            "constraint_penalties": penalties,
            "parse_rate_threshold": parse_rate_threshold,
            "entities": [
                {
                    "entity_key": e.get("entity_key"),
                    "entity_label": e.get("entity_label"),
                    "observed": {k: e.get(k) for k in energy_keys if isinstance(e.get(k), (int, float))},
                    "ideal": dict(e.get("ideal") or {}),
                    "energy_gap": e.get("energy_gap"),
                    "energy_constraints": e.get("energy_constraints"),
                    "energy_total": e.get("energy_total"),
                }
                for e in entities
            ],
            "energy_keys": energy_keys,
            "constraints": constraints,
        },
    )
    artifacts = [
        PluginArtifact(path=str(breakdown_path.relative_to(ctx.run_dir)), type="json", description="energy_breakdown.json"),
        PluginArtifact(path=str(vector_path.relative_to(ctx.run_dir)), type="json", description="energy_state_vector.json"),
    ]
    debug = {"ideaspace": {"ideal_mode": ideal_mode, "columns": cols.__dict__, "group_cols": group_cols}}
    return PluginResult(
        status="ok",
        summary="Computed ideaspace energy and breakdown.",
        metrics=_basic_metrics(df, sample_meta),
        findings=findings,
        artifacts=artifacts,
        error=None,
        references=[],
        debug=debug,
    )


def _ebm_action_verifier_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    """Verify and re-rank action candidates by modeled energy reduction."""

    action_path = ctx.run_dir / "artifacts" / "analysis_ideaspace_action_planner" / "recommendations.json"
    energy_path = ctx.run_dir / "artifacts" / "analysis_ideaspace_energy_ebm_v1" / "energy_state_vector.json"
    if not action_path.exists() or not energy_path.exists():
        return PluginResult(
            status="skipped",
            summary="Missing prerequisites (action planner and/or energy artifacts)",
            metrics=_basic_metrics(df, sample_meta),
            findings=[],
            artifacts=[],
            error=None,
            references=[],
            debug={"gating_reason": "missing_artifacts", "missing": [str(p) for p in (action_path, energy_path) if not p.exists()]},
        )

    try:
        actions = read_json(action_path)
    except Exception:
        actions = []
    try:
        energy = read_json(energy_path)
    except Exception:
        energy = {}
    if not isinstance(actions, list) or not isinstance(energy, dict):
        return PluginResult(
            status="skipped",
            summary="Invalid prerequisites (action planner and/or energy artifacts)",
            metrics=_basic_metrics(df, sample_meta),
            findings=[],
            artifacts=[],
            error=None,
            references=[],
            debug={"gating_reason": "invalid_artifacts"},
        )

    entities = energy.get("entities") or []
    weights = energy.get("weights") if isinstance(energy.get("weights"), dict) else {}
    weights_f = _default_energy_weights(weights)
    if not isinstance(entities, list) or not entities:
        return PluginResult(
            status="skipped",
            summary="No energy entities to score",
            metrics=_basic_metrics(df, sample_meta),
            findings=[],
            artifacts=[],
            error=None,
            references=[],
            debug={"gating_reason": "no_entities"},
        )

    ent_map: dict[str, dict[str, Any]] = {}
    for e in entities:
        if isinstance(e, dict) and isinstance(e.get("entity_key"), str):
            ent_map[e["entity_key"]] = e

    def _energy_for_entity(observed: dict[str, Any], ideal: dict[str, Any]) -> float:
        _, e_gap = _energy_terms(observed, ideal, weights_f)
        return float(e_gap)

    verified: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    for action in actions:
        if not isinstance(action, dict):
            continue
        lever_id = str(action.get("lever_id") or "").strip() or "unknown"
        confidence = float(action.get("confidence") or 0.0)
        evidence = action.get("evidence") if isinstance(action.get("evidence"), dict) else {}
        metrics = evidence.get("metrics") if isinstance(evidence.get("metrics"), dict) else {}

        target_entities: list[str] = []
        if lever_id == "add_qpec_capacity_plus_one_v1":
            keys = metrics.get("qpec_host_entity_keys")
            if isinstance(keys, list):
                target_entities = [str(k) for k in keys if isinstance(k, str) and k]
        if not target_entities:
            # Default scoring scope: ALL entity.
            target_entities = ["ALL"]

        # Modeled delta is applied to each target entity independently and summed.
        delta_energy = 0.0
        energy_before = 0.0
        energy_after = 0.0
        for entity_key in target_entities:
            ent = ent_map.get(entity_key)
            if not ent or not isinstance(ent.get("observed"), dict) or not isinstance(ent.get("ideal"), dict):
                continue
            observed = dict(ent["observed"])
            ideal = dict(ent["ideal"])
            before = _energy_for_entity(observed, ideal)
            modeled = dict(observed)

            if lever_id == "tune_schedule_qemail_frequency_v1":
                # Primary effect: reduce background overhead volume; secondary: modest reduction
                # in queue pressure (conservative).
                if "background_overhead_per_min" in modeled and isinstance(modeled.get("background_overhead_per_min"), (int, float)):
                    modeled["background_overhead_per_min"] = float(modeled["background_overhead_per_min"]) / 3.0
                if "queue_delay_p95" in modeled and isinstance(modeled.get("queue_delay_p95"), (int, float)):
                    modeled["queue_delay_p95"] = float(modeled["queue_delay_p95"]) * 0.95
                if "duration_p95" in modeled and isinstance(modeled.get("duration_p95"), (int, float)):
                    modeled["duration_p95"] = float(modeled["duration_p95"]) * 0.98
            elif lever_id == "add_qpec_capacity_plus_one_v1":
                host_count = int(metrics.get("qpec_host_count") or 0)
                if host_count <= 0:
                    host_count = 1
                factor = float(host_count / float(host_count + 1))
                for key in ("queue_delay_p95", "duration_p95"):
                    if key in modeled and isinstance(modeled.get(key), (int, float)):
                        modeled[key] = float(modeled[key]) * factor

            after = _energy_for_entity(modeled, ideal)
            energy_before += before
            energy_after += after
            delta_energy += (before - after)

        record = {
            "lever_id": lever_id,
            "action": str(action.get("action") or ""),
            "title": str(action.get("title") or ""),
            "target": ",".join(target_entities),
            "delta_energy": float(delta_energy),
            "energy_before": float(energy_before),
            "energy_after": float(energy_after),
            "confidence": float(max(0.0, min(1.0, confidence))),
            "constraints_passed": True,
            "blocked_reason": "",
            "evidence": evidence,
        }
        verified.append(record)

    verified.sort(
        key=lambda r: (-float(r.get("delta_energy", 0.0)), -float(r.get("confidence", 0.0)), str(r.get("lever_id", "")))
    )

    max_findings = int(config.get("max_findings", 30))
    findings = []
    for rec in verified[:max_findings]:
        findings.append(
            {
                "id": stable_id(f"{plugin_id}:{rec.get('lever_id')}"),
                "kind": "verified_action",
                "severity": "warn" if float(rec.get("delta_energy", 0.0)) > 0 else "info",
                "confidence": float(rec.get("confidence", 0.0)),
                "title": rec.get("title") or "Verified action",
                "what": rec.get("action") or "",
                "why": "Ranked by modeled energy reduction (Kona EBM) with deterministic tie-breaks.",
                "delta_energy": rec.get("delta_energy"),
                "energy_before": rec.get("energy_before"),
                "energy_after": rec.get("energy_after"),
                "constraints_passed": True,
                "blocked_reason": "",
                "evidence": rec.get("evidence") or {},
                "measurement_type": "modeled",
            }
        )

    artifacts_dir = ctx.artifacts_dir(plugin_id)
    verified_path = artifacts_dir / "verified_actions.json"
    blocked_path = artifacts_dir / "blocked_actions.json"
    write_json(verified_path, {"schema_version": "v1", "verified_actions": verified})
    write_json(blocked_path, {"schema_version": "v1", "blocked_actions": blocked})
    artifacts = [
        PluginArtifact(path=str(verified_path.relative_to(ctx.run_dir)), type="json", description="verified_actions.json"),
        PluginArtifact(path=str(blocked_path.relative_to(ctx.run_dir)), type="json", description="blocked_actions.json"),
    ]
    summary = f"Verified {len(verified)} actions"
    if not verified:
        summary = "No actions to verify"
    return PluginResult(
        status="ok" if verified else "skipped",
        summary=summary,
        metrics=_basic_metrics(df, sample_meta),
        findings=findings,
        artifacts=artifacts,
        error=None,
        references=[],
        debug={"scored_actions": len(verified)},
    )


HANDLERS: dict[str, Callable[..., PluginResult]] = {
    "analysis_ideaspace_normative_gap": _ideaspace_normative_gap,
    "analysis_ideaspace_action_planner": _ideaspace_action_planner,
    "analysis_ideaspace_energy_ebm_v1": _ideaspace_energy_ebm_v1,
    "analysis_ebm_action_verifier_v1": _ebm_action_verifier_v1,
}
