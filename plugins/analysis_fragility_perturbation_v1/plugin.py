from __future__ import annotations
import json
import logging
import math
import traceback
from pathlib import Path
from typing import Any

from statistic_harness.core.types import PluginResult, PluginError

logger = logging.getLogger(__name__)


class Plugin:
    def run(self, ctx) -> PluginResult:
        try:
            # Read EBM energy state vector artifact
            vector_path = ctx.run_dir / "artifacts" / "analysis_ideaspace_energy_ebm_v1" / "energy_state_vector.json"
            if not vector_path.exists():
                return PluginResult("na", "No energy state vector found (EBM not run)", {}, [], [], None)

            with open(vector_path) as f:
                state = json.load(f)

            entities = state.get("entities", [])
            energy_keys = state.get("energy_keys", [])
            weights = state.get("weights", {})

            if not entities or not energy_keys:
                return PluginResult("na", "Empty energy state", {}, [], [], None)

            # Focus on low-energy entities (well-performing) — fragility = which metric degradation hurts most
            findings = []
            degradation_pct = 0.20  # 20% degradation

            for entity in entities:
                observed = entity.get("observed", {})
                ideal = entity.get("ideal", {})
                base_energy = entity.get("energy_total")
                entity_key = entity.get("entity_key", "unknown")

                if base_energy is None or not math.isfinite(base_energy):
                    continue

                perturbations = []
                for metric in energy_keys:
                    obs_val = observed.get(metric)
                    ideal_val = ideal.get(metric)
                    w = weights.get(metric, 0.0)
                    if not (isinstance(obs_val, (int, float)) and isinstance(ideal_val, (int, float)) and w > 0):
                        continue

                    # Perturb: degrade observed by 20%
                    perturbed_obs = dict(observed)
                    # For minimize keys (higher is worse), increase by 20%
                    # For maximize keys (lower is worse), decrease by 20%
                    if obs_val >= ideal_val:  # minimize direction
                        perturbed_obs[metric] = obs_val * (1 + degradation_pct)
                    else:  # maximize direction
                        perturbed_obs[metric] = obs_val * (1 - degradation_pct)

                    # Recompute energy gap
                    perturbed_energy = _compute_energy(perturbed_obs, ideal, weights)
                    delta_e = perturbed_energy - base_energy

                    if delta_e > 1e-9:
                        perturbations.append({
                            "metric": metric,
                            "delta_energy": round(delta_e, 6),
                            "base_value": round(float(obs_val), 6),
                            "perturbed_value": round(float(perturbed_obs[metric]), 6),
                        })

                if perturbations:
                    perturbations.sort(key=lambda p: -p["delta_energy"])
                    findings.append({
                        "kind": "fragility",
                        "measurement_type": "measured",
                        "entity_key": entity_key,
                        "base_energy": round(base_energy, 6),
                        "most_fragile_metric": perturbations[0]["metric"],
                        "max_delta_energy": perturbations[0]["delta_energy"],
                        "perturbations": perturbations[:5],  # top 5
                        "interpretation": (
                            f"Entity {entity_key}: most fragile to {perturbations[0]['metric']} "
                            f"(\u0394E={perturbations[0]['delta_energy']:.4f} on 20% degradation)"
                        ),
                    })

            if not findings:
                return PluginResult("na", "No perturbation sensitivity detected", {}, [], [], None)

            return PluginResult(
                status="ok",
                summary=f"Fragility analysis: {len(findings)} entities analyzed, most fragile metric varies by entity",
                metrics={
                    "n_entities_analyzed": len(findings),
                    "degradation_pct": degradation_pct,
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error("Fragility analysis failed: %s", e, exc_info=True)
            return PluginResult(
                "error", f"Fragility analysis failed: {e}", {}, [], [],
                PluginError(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()),
            )


def _compute_energy(
    observed: dict[str, Any], ideal: dict[str, Any], weights: dict[str, float], eps: float = 1e-9,
) -> float:
    total = 0.0
    for metric, w in weights.items():
        if w <= 0:
            continue
        cur = observed.get(metric)
        ref = ideal.get(metric)
        if not (isinstance(cur, (int, float)) and isinstance(ref, (int, float))):
            continue
        cur_f, ref_f = float(cur), float(ref)
        if not (math.isfinite(cur_f) and math.isfinite(ref_f)):
            continue
        denom = max(abs(ref_f), abs(cur_f), 1.0, eps)
        gap = max(0.0, abs(cur_f - ref_f) / denom)
        total += w * (gap ** 2)
    return total
