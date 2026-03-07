from __future__ import annotations
import json
import logging
import math
import traceback
from typing import Any

from statistic_harness.core.types import PluginResult, PluginError

logger = logging.getLogger(__name__)


class Plugin:
    def run(self, ctx) -> PluginResult:
        try:
            vector_path = ctx.run_dir / "artifacts" / "analysis_ideaspace_energy_ebm_v1" / "energy_state_vector.json"
            if not vector_path.exists():
                return PluginResult("na", "No energy state vector found", {}, [], [], None)

            with open(vector_path) as f:
                state = json.load(f)

            entities = state.get("entities", [])
            energy_keys = state.get("energy_keys", [])

            # Need at least 2 non-ALL entities to compare
            comparable = [e for e in entities if e.get("entity_key") != "ALL"]
            if len(comparable) < 2 or not energy_keys:
                return PluginResult("na", "Need at least 2 non-ALL entities", {}, [], [], None)

            findings = []
            # Compare each pair: find entities similar on most metrics but differing on one
            for i, ea in enumerate(comparable):
                for eb in comparable[i + 1:]:
                    obs_a = ea.get("observed", {})
                    obs_b = eb.get("observed", {})

                    # Count metrics where both have values
                    shared_keys = [k for k in energy_keys
                                   if isinstance(obs_a.get(k), (int, float))
                                   and isinstance(obs_b.get(k), (int, float))]
                    if len(shared_keys) < 2:
                        continue

                    # Compute per-metric relative difference
                    diffs = {}
                    for k in shared_keys:
                        va, vb = float(obs_a[k]), float(obs_b[k])
                        denom = max(abs(va), abs(vb), 1.0)
                        diffs[k] = abs(va - vb) / denom

                    # "Similar on most" = all but 1 metric within 15% relative difference
                    threshold = 0.15
                    dissimilar = [(k, diffs[k]) for k in shared_keys if diffs[k] > threshold]
                    similar = [(k, diffs[k]) for k in shared_keys if diffs[k] <= threshold]

                    if len(dissimilar) != 1 or len(similar) < 1:
                        continue

                    diff_metric = dissimilar[0][0]
                    val_a = float(obs_a[diff_metric])
                    val_b = float(obs_b[diff_metric])

                    # Determine which entity is better on the differing metric
                    ideal_a = ea.get("ideal", {}).get(diff_metric)
                    ideal_b = eb.get("ideal", {}).get(diff_metric)

                    # Better = closer to ideal
                    if ideal_a is not None and isinstance(ideal_a, (int, float)):
                        dist_a = abs(val_a - float(ideal_a))
                        dist_b = abs(val_b - float(ideal_a))
                    else:
                        # Default: lower is better for minimize metrics
                        dist_a = val_a
                        dist_b = val_b

                    if dist_a < dist_b:
                        better, worse = ea, eb
                        better_val, worse_val = val_a, val_b
                    else:
                        better, worse = eb, ea
                        better_val, worse_val = val_b, val_a

                    findings.append({
                        "kind": "transfer_recommendation",
                        "measurement_type": "measured",
                        "source_entity": better.get("entity_key"),
                        "target_entity": worse.get("entity_key"),
                        "transfer_metric": diff_metric,
                        "source_value": round(better_val, 6),
                        "target_value": round(worse_val, 6),
                        "n_similar_metrics": len(similar),
                        "similar_metrics": [s[0] for s in similar],
                        "interpretation": (
                            f"Entity {better.get('entity_key')} achieves {diff_metric}="
                            f"{better_val:.4f} vs {worse.get('entity_key')} at {worse_val:.4f}. "
                            f"Entities are similar on {len(similar)} other metrics — "
                            f"transfer {diff_metric} practices from {better.get('entity_key')} to {worse.get('entity_key')}."
                        ),
                    })

            if not findings:
                return PluginResult("na", "No transfer opportunities found", {}, [], [], None)

            # Sort by magnitude of difference
            findings.sort(key=lambda f: abs(f["source_value"] - f["target_value"]), reverse=True)

            return PluginResult(
                status="ok",
                summary=f"Cross-entity transfer: {len(findings)} transfer opportunities identified",
                metrics={"n_transfer_opportunities": len(findings), "n_entities_compared": len(comparable)},
                findings=findings[:10],  # top 10
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error("Cross-entity transfer failed: %s", e, exc_info=True)
            return PluginResult(
                "error", f"Cross-entity transfer failed: {e}", {}, [], [],
                PluginError(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()),
            )
