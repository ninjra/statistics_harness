"""Static dispatch table mapping analysis finding kinds to lever templates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LeverTemplate:
    lever_id: str
    title: str
    action_template: str
    confidence_base: float


FINDING_LEVER_MAP: list[dict[str, Any]] = [
    {
        "finding_kind": "granger_causal_link",
        "min_confidence": 0.05,  # p_value threshold
        "lever": LeverTemplate(
            lever_id="decouple_causal_dependency",
            title="Decouple causally linked processes",
            action_template="Process {cause} Granger-causes delays in {effect} (p={p_value:.4f}). Decouple by buffering, async handoff, or scheduling separation.",
            confidence_base=0.55,
        ),
    },
    {
        "finding_kind": "changepoint_detected",
        "min_magnitude": 1.5,
        "lever": LeverTemplate(
            lever_id="investigate_regime_change",
            title="Investigate detected regime change",
            action_template="Changepoint detected with magnitude {magnitude:.2f}. Investigate root cause (deployment, config, volume shift).",
            confidence_base=0.50,
        ),
    },
    {
        "finding_kind": "anomaly_cluster",
        "min_anomaly_rate": 0.05,
        "lever": LeverTemplate(
            lever_id="address_anomaly_hotspot",
            title="Address anomaly hotspot",
            action_template="Anomaly rate of {anomaly_rate:.1%} detected. Root-cause: configuration drift, data quality, or resource contention.",
            confidence_base=0.45,
        ),
    },
    {
        "finding_kind": "hidden_markov_regime",
        "min_states": 2,
        "lever": LeverTemplate(
            lever_id="stabilize_regime_oscillation",
            title="Stabilize process regime oscillation",
            action_template="HMM detected {n_states} distinct operating regimes. Stabilize: pin config during close windows, add circuit breaker to prevent regime transitions under load.",
            confidence_base=0.50,
        ),
    },
    {
        "finding_kind": "transfer_entropy_link",
        "min_te": 0.1,
        "lever": LeverTemplate(
            lever_id="break_information_flow_bottleneck",
            title="Break information-flow bottleneck between processes",
            action_template="Process {source} drives {target} with TE={transfer_entropy:.3f}. Add buffering or decouple to prevent cascading delays.",
            confidence_base=0.50,
        ),
    },
    {
        "finding_kind": "spectral_clustering_community",
        "min_modularity": 0.3,
        "lever": LeverTemplate(
            lever_id="isolate_process_community",
            title="Isolate tightly-coupled process community onto dedicated resources",
            action_template="Tightly-coupled community detected (modularity={modularity:.2f}). Isolate onto dedicated resources to contain blast radius.",
            confidence_base=0.50,
        ),
    },
    {
        "finding_kind": "survival_hazard_spike",
        "min_hazard_ratio": 2.0,
        "lever": LeverTemplate(
            lever_id="mitigate_high_hazard_process",
            title="Mitigate high-hazard process timeout risk",
            action_template="Hazard ratio {hazard_ratio:.1f}x — high risk of exceeding SLA. Add proactive timeout, early warning, or fallback path.",
            confidence_base=0.55,
        ),
    },
    {
        "finding_kind": "causal",
        "min_effect": 0.0,
        "lever": LeverTemplate(
            lever_id="address_causal_treatment_effect",
            title="Address significant causal treatment effect",
            action_template="Causal analysis detected significant treatment effect (ATE={ate:+.4f}). Investigate whether to amplify or mitigate.",
            confidence_base=0.50,
        ),
    },
    {
        "finding_kind": "counterfactual",
        "min_effect": 0.0,
        "lever": LeverTemplate(
            lever_id="investigate_counterfactual_impact",
            title="Investigate counterfactual impact",
            action_template="Counterfactual analysis detected impact. Review intervention timing and effect magnitude.",
            confidence_base=0.45,
        ),
    },
    {
        "finding_kind": "role_inference",
        "min_importance": 0.0,
        "lever": LeverTemplate(
            lever_id="optimize_key_driver",
            title="Optimize identified key driver",
            action_template="Feature attribution identified key drivers. Focus optimization on highest-impact features.",
            confidence_base=0.40,
        ),
    },
]


def match_finding_to_levers(finding: dict[str, Any]) -> list[dict[str, Any]]:
    """Match a single finding to applicable lever templates.

    Returns list of dicts with 'lever' (LeverTemplate) and 'finding' keys.
    """
    kind = str(finding.get("kind", ""))
    matches = []
    for entry in FINDING_LEVER_MAP:
        if entry["finding_kind"] != kind:
            continue
        matches.append({"lever": entry["lever"], "finding": finding, "finding_kind": kind})
    return matches
