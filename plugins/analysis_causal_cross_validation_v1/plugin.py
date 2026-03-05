from __future__ import annotations
import logging
import traceback
import json
from statistic_harness.core.types import PluginResult, PluginArtifact, PluginError

logger = logging.getLogger(__name__)

# Causal plugin IDs whose results we cross-validate
CAUSAL_PLUGINS = [
    "analysis_double_ml_ate_v1",
    "analysis_diff_in_diff_v1",
    "analysis_propensity_score_matching_v1",
    "analysis_inverse_propensity_weighting_v1",
    "analysis_meta_learner_cate_v1",
    "analysis_causal_forest_hte_v1",
    "analysis_synthetic_control_v1",
]

EFFECT_KEYS = ["ate", "did_estimate", "att", "ipw_ate", "mean_treatment_effect"]


def _extract_effect(result_data: dict) -> float | None:
    """Extract the primary effect estimate from a plugin result."""
    metrics = result_data.get("metrics") or {}
    if isinstance(metrics, str):
        try:
            metrics = json.loads(metrics)
        except Exception:
            metrics = {}
    for key in EFFECT_KEYS:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, (int, float)):
                return float(val)
    # Try findings
    findings = result_data.get("findings") or []
    if isinstance(findings, str):
        try:
            findings = json.loads(findings)
        except Exception:
            findings = []
    for f in findings:
        if not isinstance(f, dict):
            continue
        for key in EFFECT_KEYS:
            if key in f:
                val = f[key]
                if isinstance(val, (int, float)):
                    return float(val)
    return None


class Plugin:
    def run(self, ctx) -> PluginResult:
        try:
            # Read upstream causal plugin results from storage
            effects: dict[str, float] = {}
            for pid in CAUSAL_PLUGINS:
                try:
                    result_data = ctx.storage.fetch_latest_plugin_result(
                        ctx.run_id, pid
                    )
                    if not result_data:
                        continue
                    status = str(result_data.get("status") or "").lower()
                    if status != "ok":
                        continue
                    effect = _extract_effect(result_data)
                    if effect is not None:
                        effects[pid] = effect
                except Exception:
                    continue

            if len(effects) < 2:
                return PluginResult(
                    "ok",
                    f"Cross-validation: only {len(effects)} causal plugins produced effects, need >=2",
                    {"n_causal_plugins_with_effects": len(effects)},
                    [{
                        "kind": "causal",
                        "measurement_type": "not_applicable",
                        "reason": "insufficient_upstream_results",
                    }],
                    [], None,
                )

            # Check 1: Sign agreement
            signs = {pid: (1 if e > 0 else -1 if e < 0 else 0) for pid, e in effects.items()}
            positive = sum(1 for s in signs.values() if s > 0)
            negative = sum(1 for s in signs.values() if s < 0)
            total = len(signs)
            sign_agreement = max(positive, negative) / total
            sign_disagreement = min(positive, negative)

            # Check 2: Magnitude agreement (max/min ratio)
            abs_effects = [abs(e) for e in effects.values() if e != 0]
            if len(abs_effects) >= 2:
                magnitude_ratio = max(abs_effects) / max(min(abs_effects), 1e-10)
            else:
                magnitude_ratio = 1.0

            # Synthesis
            findings = []

            if sign_disagreement > 0:
                pos_plugins = [p for p, s in signs.items() if s > 0]
                neg_plugins = [p for p, s in signs.items() if s < 0]
                findings.append({
                    "kind": "causal",
                    "measurement_type": "measured",
                    "check": "sign_disagreement",
                    "sign_agreement_ratio": round(sign_agreement, 4),
                    "positive_effect_plugins": pos_plugins,
                    "negative_effect_plugins": neg_plugins,
                    "severity": "warning",
                    "interpretation": (
                        f"Sign disagreement: {positive} plugins estimate positive effect, "
                        f"{negative} estimate negative. Review assumptions."
                    ),
                })

            if magnitude_ratio > 3.0:
                findings.append({
                    "kind": "causal",
                    "measurement_type": "measured",
                    "check": "magnitude_disagreement",
                    "magnitude_ratio": round(magnitude_ratio, 2),
                    "effects": {pid: round(e, 6) for pid, e in effects.items()},
                    "severity": "warning",
                    "interpretation": (
                        f"Magnitude disagreement: effect estimates vary by {magnitude_ratio:.1f}x. "
                        f"Consider which method's assumptions best fit the data."
                    ),
                })

            # Synthesis finding
            majority_sign = "positive" if positive >= negative else "negative"
            effect_values = list(effects.values())
            median_effect = float(sorted(effect_values)[len(effect_values) // 2])
            findings.append({
                "kind": "causal",
                "measurement_type": "measured",
                "check": "synthesis",
                "n_methods": total,
                "sign_agreement_ratio": round(sign_agreement, 4),
                "magnitude_ratio": round(magnitude_ratio, 2),
                "median_effect": round(median_effect, 6),
                "majority_direction": majority_sign,
                "all_effects": {pid: round(e, 6) for pid, e in effects.items()},
                "interpretation": (
                    f"{int(sign_agreement * 100)}% of {total} causal methods agree on "
                    f"{majority_sign} effect (median={median_effect:+.4f})"
                ),
            })

            return PluginResult(
                status="ok",
                summary=(
                    f"Cross-validation: {total} methods, {sign_agreement:.0%} sign agreement, "
                    f"median effect={median_effect:+.4f}"
                ),
                metrics={
                    "n_causal_plugins_with_effects": total,
                    "sign_agreement_ratio": round(sign_agreement, 4),
                    "magnitude_ratio": round(magnitude_ratio, 2),
                    "median_effect": round(median_effect, 6),
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error("Cross-validation failed: %s", e, exc_info=True)
            return PluginResult("error", f"Cross-validation failed: {e}", {}, [], [],
                PluginError(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()))
