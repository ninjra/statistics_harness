from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class RouteCandidate:
    lever_id: str
    title: str
    action: str
    action_type: str
    confidence: float
    target_entity_keys: tuple[str, ...]
    target_process_ids: tuple[str, ...]
    evidence_metrics: dict[str, Any]
    # Single-step reference (used only for sorting/heuristics; must not be trusted as multi-step truth):
    delta_energy_single: float
    energy_before_single: float


@dataclass(frozen=True)
class RouteConfig:
    route_max_depth: int
    route_beam_width: int
    route_min_delta_energy: float
    route_min_confidence: float
    route_allow_cross_target_steps: bool
    route_stop_energy_threshold: float
    route_candidate_limit: int
    route_time_budget_ms: int
    route_disallowed_lever_ids: tuple[str, ...]
    route_disallowed_action_types: tuple[str, ...]


@dataclass(frozen=True)
class RouteStep:
    step_index: int
    lever_id: str
    action_type: str
    title: str
    action: str
    confidence: float
    target_entity_keys: tuple[str, ...]
    target_process_ids: tuple[str, ...]
    energy_before: float
    energy_after: float
    delta_energy: float
    modeled_metrics_after: dict[str, float]


@dataclass
class _RouteState:
    metrics_by_entity: dict[str, dict[str, float]]
    ideal_by_entity: dict[str, dict[str, float]]
    constraints_by_entity: dict[str, float]
    scope_entity_keys: tuple[str, ...]
    route_energy_before: float
    route_energy_after: float
    steps: list[RouteStep]
    confidence_product: float
    confidence_count: int
    signature: str


def _clamp_confidence(value: Any) -> float:
    if not isinstance(value, (int, float)):
        return 0.0
    val = float(value)
    if not math.isfinite(val):
        return 0.0
    return max(0.0, min(1.0, val))


def _to_float(value: Any, default: float = 0.0) -> float:
    if not isinstance(value, (int, float)):
        return float(default)
    val = float(value)
    if not math.isfinite(val):
        return float(default)
    return val


def _clean_metrics(raw: Mapping[str, Any] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(raw, Mapping):
        return out
    for key in sorted(str(k) for k in raw.keys()):
        value = raw.get(key)
        if isinstance(value, (int, float)):
            f = float(value)
            if math.isfinite(f):
                out[key] = f
    return out


def _normalize_targets(values: Sequence[str] | None) -> tuple[str, ...]:
    out: list[str] = []
    for value in values or ():
        token = str(value or "").strip()
        if token:
            out.append(token)
    if not out:
        return ("ALL",)
    return tuple(out)


def _target_signature(values: Sequence[str]) -> str:
    uniq = sorted({str(v).strip() for v in values if str(v).strip()})
    return ",".join(uniq)


def _ordered_union(left: Sequence[str], right: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for bucket in (left, right):
        for value in bucket:
            token = str(value).strip()
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)
    return tuple(out)


def _route_confidence(state: _RouteState) -> float:
    if state.confidence_count <= 0:
        return 0.0
    base = max(0.0, state.confidence_product)
    return float(base ** (1.0 / float(state.confidence_count)))


def _state_total_delta(state: _RouteState) -> float:
    return float(max(0.0, state.route_energy_before - state.route_energy_after))


def _state_sort_key(state: _RouteState) -> tuple[float, float, int, str]:
    return (
        -_state_total_delta(state),
        -_route_confidence(state),
        len(state.steps),
        state.signature,
    )


def _candidate_sort_key(candidate: RouteCandidate) -> tuple[float, float, str, str]:
    return (
        -_to_float(candidate.delta_energy_single, 0.0),
        -_clamp_confidence(candidate.confidence),
        str(candidate.lever_id),
        ",".join(candidate.target_process_ids),
    )


def _scope_energy(
    metrics_by_entity: Mapping[str, Mapping[str, float]],
    ideal_by_entity: Mapping[str, Mapping[str, float]],
    constraints_by_entity: Mapping[str, float],
    scope_entity_keys: Sequence[str],
    weights: Mapping[str, float],
    energy_gap: Callable[[Mapping[str, float], Mapping[str, float], Mapping[str, float]], float],
) -> float:
    total = 0.0
    for entity_key in scope_entity_keys:
        observed = metrics_by_entity.get(entity_key)
        ideal = ideal_by_entity.get(entity_key)
        if not isinstance(observed, Mapping) or not isinstance(ideal, Mapping):
            continue
        gap = _to_float(energy_gap(observed, ideal, weights), 0.0)
        constraints = _to_float(constraints_by_entity.get(entity_key), 0.0)
        total += max(0.0, gap + constraints)
    return float(max(0.0, total))


def _aggregate_metrics(
    metrics_by_entity: Mapping[str, Mapping[str, float]],
    target_entity_keys: Sequence[str],
) -> dict[str, float]:
    keys: set[str] = set()
    rows: list[Mapping[str, float]] = []
    for entity_key in target_entity_keys:
        row = metrics_by_entity.get(entity_key)
        if isinstance(row, Mapping):
            rows.append(row)
            keys.update(str(k) for k in row.keys())
    out: dict[str, float] = {}
    for key in sorted(keys):
        vals: list[float] = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                vals.append(float(value))
        if vals:
            out[key] = float(sum(vals) / float(len(vals)))
    return out


def _config_snapshot(config: RouteConfig) -> dict[str, Any]:
    return {
        "route_max_depth": int(config.route_max_depth),
        "route_beam_width": int(config.route_beam_width),
        "route_min_delta_energy": float(config.route_min_delta_energy),
        "route_min_confidence": float(config.route_min_confidence),
        "route_allow_cross_target_steps": bool(config.route_allow_cross_target_steps),
        "route_stop_energy_threshold": float(config.route_stop_energy_threshold),
        "route_candidate_limit": int(config.route_candidate_limit),
        "route_time_budget_ms": int(config.route_time_budget_ms),
        "route_disallowed_lever_ids": list(config.route_disallowed_lever_ids),
        "route_disallowed_action_types": list(config.route_disallowed_action_types),
    }


def _not_applicable(
    *,
    plugin_id: str,
    generated_at: str,
    config: RouteConfig,
    reason_code: str,
    message: str,
    expanded_states: int,
    details: dict[str, Any] | None = None,
    debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "kona_route_plan.v1",
        "plugin_id": plugin_id,
        "generated_at": generated_at,
        "decision": "not_applicable",
        "target_signature": None,
        "config": _config_snapshot(config),
        "not_applicable": {
            "reason_code": str(reason_code),
            "message": str(message),
            "details": details or {},
        },
        "steps": [],
        "totals": {
            "energy_before": 0.0,
            "energy_after": 0.0,
            "total_delta_energy": 0.0,
            "route_confidence": 0.0,
            "stop_reason": str(reason_code).strip().lower() or "not_applicable",
            "expanded_states": int(max(0, expanded_states)),
        },
        "debug": debug or {},
    }


def _budget_exceeded(config: RouteConfig, start_ms: float, time_now_ms: Callable[[], float]) -> bool:
    budget_ms = int(config.route_time_budget_ms)
    if budget_ms <= 0:
        return False
    elapsed = float(time_now_ms()) - float(start_ms)
    return elapsed >= float(budget_ms)


def _clone_metrics_by_entity(source: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key in sorted(str(k) for k in source.keys()):
        row = source.get(key)
        if not isinstance(row, Mapping):
            continue
        out[key] = {str(mk): float(mv) for mk, mv in row.items() if isinstance(mv, (int, float))}
    return out


def _expand_state(
    *,
    state: _RouteState,
    candidate: RouteCandidate,
    config: RouteConfig,
    base_metrics_by_entity: Mapping[str, Mapping[str, float]],
    base_constraints_by_entity: Mapping[str, float],
    weights: Mapping[str, float],
    apply_lever: Callable[[str, Mapping[str, float], Mapping[str, Any]], dict[str, float]],
    energy_gap: Callable[[Mapping[str, float], Mapping[str, float], Mapping[str, float]], float],
    update_constraints: Callable[[str, Mapping[str, float], Mapping[str, float], float, Mapping[str, Any]], float],
) -> _RouteState | None:
    target_entity_keys = _normalize_targets(candidate.target_entity_keys)
    for entity_key in target_entity_keys:
        if entity_key not in state.metrics_by_entity or entity_key not in state.ideal_by_entity:
            return None

    if state.steps and not config.route_allow_cross_target_steps:
        if _target_signature(target_entity_keys) != _target_signature(state.scope_entity_keys):
            return None
        scope_entity_keys = tuple(state.scope_entity_keys)
    elif not state.steps:
        scope_entity_keys = target_entity_keys
    else:
        scope_entity_keys = _ordered_union(state.scope_entity_keys, target_entity_keys)

    step_signature = f"{candidate.lever_id}@{_target_signature(target_entity_keys)}"
    if state.signature:
        existing = set(state.signature.split("|"))
        if step_signature in existing:
            return None
        signature = f"{state.signature}|{step_signature}"
    else:
        signature = step_signature

    metrics_by_entity = _clone_metrics_by_entity(state.metrics_by_entity)
    constraints_by_entity = {
        key: _to_float(value, 0.0) for key, value in state.constraints_by_entity.items()
    }

    energy_before = _scope_energy(
        metrics_by_entity,
        state.ideal_by_entity,
        constraints_by_entity,
        scope_entity_keys,
        weights,
        energy_gap,
    )

    evidence_metrics = dict(candidate.evidence_metrics or {})
    for entity_key in target_entity_keys:
        before_metrics = dict(metrics_by_entity[entity_key])
        after_metrics_raw = apply_lever(candidate.lever_id, before_metrics, evidence_metrics) or {}
        after_metrics = _clean_metrics(after_metrics_raw)
        if not after_metrics:
            after_metrics = before_metrics
        metrics_by_entity[entity_key] = dict(after_metrics)

        prev_constraints = _to_float(constraints_by_entity.get(entity_key), 0.0)
        next_constraints = update_constraints(
            candidate.lever_id,
            before_metrics,
            after_metrics,
            prev_constraints,
            evidence_metrics,
        )
        constraints_by_entity[entity_key] = max(0.0, _to_float(next_constraints, prev_constraints))

    energy_after = _scope_energy(
        metrics_by_entity,
        state.ideal_by_entity,
        constraints_by_entity,
        scope_entity_keys,
        weights,
        energy_gap,
    )
    delta_energy = max(0.0, float(energy_before - energy_after))
    if delta_energy < float(config.route_min_delta_energy):
        return None
    if energy_after > energy_before + 1e-12:
        return None

    route_energy_before = _scope_energy(
        base_metrics_by_entity,
        state.ideal_by_entity,
        base_constraints_by_entity,
        scope_entity_keys,
        weights,
        energy_gap,
    )
    modeled_metrics_after = _aggregate_metrics(metrics_by_entity, target_entity_keys)
    confidence = _clamp_confidence(candidate.confidence)
    step = RouteStep(
        step_index=len(state.steps) + 1,
        lever_id=str(candidate.lever_id),
        action_type=str(candidate.action_type or "ideaspace_action"),
        title=str(candidate.title),
        action=str(candidate.action),
        confidence=confidence,
        target_entity_keys=tuple(target_entity_keys),
        target_process_ids=tuple(candidate.target_process_ids),
        energy_before=float(max(0.0, energy_before)),
        energy_after=float(max(0.0, energy_after)),
        delta_energy=float(delta_energy),
        modeled_metrics_after=modeled_metrics_after,
    )
    return _RouteState(
        metrics_by_entity=metrics_by_entity,
        ideal_by_entity=state.ideal_by_entity,
        constraints_by_entity=constraints_by_entity,
        scope_entity_keys=tuple(scope_entity_keys),
        route_energy_before=float(max(0.0, route_energy_before)),
        route_energy_after=float(max(0.0, energy_after)),
        steps=[*state.steps, step],
        confidence_product=float(state.confidence_product * confidence),
        confidence_count=int(state.confidence_count + 1),
        signature=signature,
    )


def solve_kona_route_plan(
    *,
    plugin_id: str,
    generated_at: str,
    entities: Mapping[str, Mapping[str, Any]],
    weights: Mapping[str, float],
    candidates: Sequence[RouteCandidate],
    config: RouteConfig,
    apply_lever: Callable[[str, Mapping[str, float], Mapping[str, Any]], dict[str, float]],
    energy_gap: Callable[[Mapping[str, float], Mapping[str, float], Mapping[str, float]], float],
    update_constraints: Callable[[str, Mapping[str, float], Mapping[str, float], float, Mapping[str, Any]], float],
    time_now_ms: Callable[[], float],
) -> dict[str, Any]:
    if int(config.route_max_depth) < 2 or int(config.route_beam_width) < 1:
        return _not_applicable(
            plugin_id=plugin_id,
            generated_at=generated_at,
            config=config,
            reason_code="ROUTE_DISABLED",
            message="Route planning is disabled by configuration.",
            expanded_states=0,
        )

    entity_metrics_by_key: dict[str, dict[str, float]] = {}
    entity_ideal_by_key: dict[str, dict[str, float]] = {}
    entity_constraints_by_key: dict[str, float] = {}
    for entity_key in sorted(str(k) for k in entities.keys()):
        payload = entities.get(entity_key)
        if not isinstance(payload, Mapping):
            continue
        observed = _clean_metrics(payload.get("observed"))
        ideal = _clean_metrics(payload.get("ideal"))
        if not observed or not ideal:
            continue
        entity_metrics_by_key[entity_key] = observed
        entity_ideal_by_key[entity_key] = ideal
        entity_constraints_by_key[entity_key] = max(0.0, _to_float(payload.get("energy_constraints"), 0.0))

    if not entity_metrics_by_key:
        return _not_applicable(
            plugin_id=plugin_id,
            generated_at=generated_at,
            config=config,
            reason_code="NO_ENTITIES",
            message="No route-eligible entities were available.",
            expanded_states=0,
        )

    disallowed_levers = {str(v).strip() for v in config.route_disallowed_lever_ids if str(v).strip()}
    disallowed_action_types = {str(v).strip() for v in config.route_disallowed_action_types if str(v).strip()}
    filtered_candidates: list[RouteCandidate] = []
    dropped: dict[str, int] = {
        "confidence": 0,
        "lever": 0,
        "action_type": 0,
    }
    for candidate in candidates:
        confidence = _clamp_confidence(candidate.confidence)
        if confidence < float(config.route_min_confidence):
            dropped["confidence"] += 1
            continue
        if str(candidate.lever_id) in disallowed_levers:
            dropped["lever"] += 1
            continue
        if str(candidate.action_type) in disallowed_action_types:
            dropped["action_type"] += 1
            continue
        target_entity_keys = _normalize_targets(candidate.target_entity_keys)
        filtered_candidates.append(
            RouteCandidate(
                lever_id=str(candidate.lever_id),
                title=str(candidate.title),
                action=str(candidate.action),
                action_type=str(candidate.action_type),
                confidence=confidence,
                target_entity_keys=target_entity_keys,
                target_process_ids=tuple(str(v).strip() for v in candidate.target_process_ids if str(v).strip()),
                evidence_metrics=dict(candidate.evidence_metrics or {}),
                delta_energy_single=_to_float(candidate.delta_energy_single, 0.0),
                energy_before_single=_to_float(candidate.energy_before_single, 0.0),
            )
        )
    filtered_candidates.sort(key=_candidate_sort_key)
    filtered_candidates = filtered_candidates[: max(1, int(config.route_candidate_limit))]

    if not filtered_candidates:
        return _not_applicable(
            plugin_id=plugin_id,
            generated_at=generated_at,
            config=config,
            reason_code="NO_CANDIDATES",
            message="No route candidates passed route filters.",
            expanded_states=0,
            details={"dropped": dropped},
        )

    base_metrics_by_entity = _clone_metrics_by_entity(entity_metrics_by_key)
    base_constraints_by_entity = dict(entity_constraints_by_key)
    state0 = _RouteState(
        metrics_by_entity=_clone_metrics_by_entity(entity_metrics_by_key),
        ideal_by_entity=entity_ideal_by_key,
        constraints_by_entity=dict(entity_constraints_by_key),
        scope_entity_keys=tuple(),
        route_energy_before=0.0,
        route_energy_after=0.0,
        steps=[],
        confidence_product=1.0,
        confidence_count=0,
        signature="",
    )

    start_ms = float(time_now_ms())
    frontier: list[_RouteState] = [state0]
    best_state: _RouteState | None = None
    expanded_states = 0
    stop_reason = "max_depth"
    timed_out = False

    for depth in range(1, int(config.route_max_depth) + 1):
        if _budget_exceeded(config, start_ms, time_now_ms):
            timed_out = True
            break
        next_states: list[_RouteState] = []
        for state in frontier:
            if _budget_exceeded(config, start_ms, time_now_ms):
                timed_out = True
                break
            for candidate in filtered_candidates:
                if _budget_exceeded(config, start_ms, time_now_ms):
                    timed_out = True
                    break
                expanded_states += 1
                expanded = _expand_state(
                    state=state,
                    candidate=candidate,
                    config=config,
                    base_metrics_by_entity=base_metrics_by_entity,
                    base_constraints_by_entity=base_constraints_by_entity,
                    weights=weights,
                    apply_lever=apply_lever,
                    energy_gap=energy_gap,
                    update_constraints=update_constraints,
                )
                if expanded is None:
                    continue
                next_states.append(expanded)
            if timed_out:
                break
        if timed_out:
            break
        if not next_states:
            stop_reason = "no_expandable_states" if best_state is None else "no_further_improvement"
            break
        next_states.sort(key=_state_sort_key)
        frontier = next_states[: max(1, int(config.route_beam_width))]
        cand_best = frontier[0]
        if best_state is None or _state_sort_key(cand_best) < _state_sort_key(best_state):
            best_state = cand_best
        if best_state is not None and best_state.route_energy_after <= float(config.route_stop_energy_threshold):
            stop_reason = "threshold_reached"
            break
        if depth == int(config.route_max_depth):
            stop_reason = "max_depth"

    if timed_out:
        if best_state is None:
            return _not_applicable(
                plugin_id=plugin_id,
                generated_at=generated_at,
                config=config,
                reason_code="TIME_BUDGET_EXCEEDED",
                message="Route search exceeded configured route_time_budget_ms before finding a route.",
                expanded_states=expanded_states,
                details={"route_time_budget_ms": int(config.route_time_budget_ms)},
            )
        stop_reason = "time_budget_exceeded"

    if best_state is None or not best_state.steps:
        return _not_applicable(
            plugin_id=plugin_id,
            generated_at=generated_at,
            config=config,
            reason_code="NO_VALID_ROUTE",
            message="No modeled multi-step route satisfied route constraints.",
            expanded_states=expanded_states,
            details={"candidate_count": len(filtered_candidates)},
        )

    total_delta = _state_total_delta(best_state)
    if total_delta < float(config.route_min_delta_energy):
        return _not_applicable(
            plugin_id=plugin_id,
            generated_at=generated_at,
            config=config,
            reason_code="ROUTE_DELTA_BELOW_MIN",
            message="Best route did not satisfy route_min_delta_energy.",
            expanded_states=expanded_states,
            details={
                "total_delta_energy": total_delta,
                "route_min_delta_energy": float(config.route_min_delta_energy),
            },
        )

    payload_steps: list[dict[str, Any]] = []
    for step in best_state.steps:
        payload_steps.append(
            {
                "step_index": int(step.step_index),
                "lever_id": step.lever_id,
                "action_type": step.action_type,
                "title": step.title,
                "action": step.action,
                "confidence": float(step.confidence),
                "target_entity_keys": list(step.target_entity_keys),
                "target_process_ids": list(step.target_process_ids),
                "energy_before": float(max(0.0, step.energy_before)),
                "energy_after": float(max(0.0, step.energy_after)),
                "delta_energy": float(max(0.0, step.delta_energy)),
                "modeled_metrics_after": dict(step.modeled_metrics_after),
            }
        )

    return {
        "schema_version": "kona_route_plan.v1",
        "plugin_id": plugin_id,
        "generated_at": generated_at,
        "decision": "modeled",
        "target_signature": _target_signature(best_state.scope_entity_keys) or None,
        "config": _config_snapshot(config),
        "not_applicable": None,
        "steps": payload_steps,
        "totals": {
            "energy_before": float(max(0.0, best_state.route_energy_before)),
            "energy_after": float(max(0.0, best_state.route_energy_after)),
            "total_delta_energy": float(total_delta),
            "route_confidence": float(_route_confidence(best_state)),
            "stop_reason": stop_reason,
            "expanded_states": int(max(0, expanded_states)),
        },
        "debug": {
            "candidate_count": int(len(filtered_candidates)),
            "dropped_candidates": dropped,
            "beam_width": int(config.route_beam_width),
            "max_depth": int(config.route_max_depth),
        },
    }
