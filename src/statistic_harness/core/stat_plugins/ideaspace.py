from __future__ import annotations

from typing import Any, Callable

import math

import numpy as np
import pandas as pd

from statistic_harness.core.baselines import load_signed_baseline
from statistic_harness.core.ideaspace_feature_extractor import (
    IdeaspaceColumns,
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


HANDLERS: dict[str, Callable[..., PluginResult]] = {
    "analysis_ideaspace_normative_gap": _ideaspace_normative_gap,
    "analysis_ideaspace_action_planner": _ideaspace_action_planner,
}
