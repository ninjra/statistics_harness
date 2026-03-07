from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


def _pick_two_numeric(df: pd.DataFrame) -> tuple[str | None, str | None]:
    cols = [str(col) for col in df.select_dtypes(include="number").columns]
    if len(cols) < 2:
        return None, None
    return cols[0], cols[1]


def _get_confounders(df: pd.DataFrame, t_col: str, y_col: str) -> list[str]:
    numeric = [str(c) for c in df.select_dtypes(include="number").columns if str(c) not in (t_col, y_col)]
    return numeric[:10]


def _run_dowhy_analysis(df_analysis: pd.DataFrame, t_col: str, y_col: str, confounders: list[str], seed: int) -> dict | None:
    try:
        import dowhy
        from dowhy import CausalModel
    except ImportError:
        return None

    gml = 'graph [directed 1\n'
    all_nodes = [t_col, y_col] + confounders
    for node in all_nodes:
        gml += f'  node [id "{node}" label "{node}"]\n'
    gml += f'  edge [source "{t_col}" target "{y_col}"]\n'
    for c in confounders:
        gml += f'  edge [source "{c}" target "{t_col}"]\n'
        gml += f'  edge [source "{c}" target "{y_col}"]\n'
    gml += ']'

    try:
        model = CausalModel(
            data=df_analysis,
            treatment=t_col,
            outcome=y_col,
            graph=gml,
        )
        identified = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            identified,
            method_name="backdoor.linear_regression",
        )
        effect_value = float(estimate.value)

        refutations = []
        try:
            placebo_ref = model.refute_estimate(
                identified, estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute",
                num_simulations=50,
            )
            refutations.append({
                "name": "placebo_treatment",
                "status": "pass" if abs(float(placebo_ref.new_effect)) < abs(effect_value) * 0.5 else "warn",
                "detail": f"placebo_effect={float(placebo_ref.new_effect):.6f}",
            })
        except Exception:
            pass

        try:
            random_ref = model.refute_estimate(
                identified, estimate,
                method_name="random_common_cause",
                num_simulations=50,
            )
            refutations.append({
                "name": "random_common_cause",
                "status": "pass" if abs(float(random_ref.new_effect) - effect_value) < abs(effect_value) * 0.2 else "warn",
                "detail": f"new_effect={float(random_ref.new_effect):.6f}",
            })
        except Exception:
            pass

        return {
            "method": "dowhy_backdoor_linear_regression",
            "effect": effect_value,
            "refutations": refutations,
            "confounders_used": confounders,
        }
    except Exception:
        return None


class Plugin:
    def run(self, ctx) -> PluginResult:
        loader = getattr(ctx, "dataset_loader", None)
        if not callable(loader):
            return PluginResult("na", "dataset_loader unavailable", {}, [], [], None)
        df = loader()
        if df is None or len(df) == 0:
            return PluginResult("na", "Empty dataset", {"rows": 0}, [], [], None)

        t_col = ctx.settings.get("treatment_column")
        y_col = ctx.settings.get("outcome_column")
        if not (isinstance(t_col, str) and t_col in df.columns and isinstance(y_col, str) and y_col in df.columns):
            t_guess, y_guess = _pick_two_numeric(df)
            t_col = t_col if isinstance(t_col, str) and t_col in df.columns else t_guess
            y_col = y_col if isinstance(y_col, str) and y_col in df.columns else y_guess
        if not t_col or not y_col:
            return PluginResult(
                "na",
                "Need treatment and outcome numeric columns",
                {"rows": int(len(df)), "columns": int(len(df.columns))},
                [], [], None,
            )

        t = pd.to_numeric(df[t_col], errors="coerce")
        y = pd.to_numeric(df[y_col], errors="coerce")
        valid = t.notna() & y.notna()
        t = t[valid]
        y = y[valid]
        if len(t) < 8:
            return PluginResult(
                "na",
                "Not enough valid rows for causal approximation",
                {"rows_used": int(len(t))},
                [], [], None,
            )

        threshold = ctx.settings.get("treatment_threshold")
        if threshold is None:
            threshold = float(t.median())
        treated = t > float(threshold)
        if int(treated.sum()) == 0 or int((~treated).sum()) == 0:
            return PluginResult(
                "degraded",
                "Treatment split degenerate; effect not identified",
                {"rows_used": int(len(t))},
                [
                    {
                        "kind": "recommendation",
                        "status": "not_identified",
                        "reason": "degenerate_treatment_split",
                    }
                ],
                [], None,
            )

        seed = int(getattr(ctx, "run_seed", 0) or 0)
        confounders = _get_confounders(df, t_col, y_col)

        # Try dowhy-based causal analysis first
        df_analysis = df[[t_col, y_col] + [c for c in confounders if c in df.columns]].dropna().copy()
        dowhy_result = None
        if len(confounders) > 0 and len(df_analysis) >= 8:
            dowhy_result = _run_dowhy_analysis(df_analysis, t_col, y_col, confounders, seed)

        if dowhy_result is not None:
            effect = dowhy_result["effect"]
            method = dowhy_result["method"]
            refutations = dowhy_result["refutations"]
            assumptions = [
                "Causal graph assumes confounders are observed.",
                "Backdoor adjustment via linear regression.",
                "No unmeasured confounding conditional on observed covariates.",
            ]
        else:
            # Fallback: association analysis with explicit disclaimer
            effect = float(y[treated].mean() - y[~treated].mean())
            method = "association_difference_in_means"
            rng = np.random.default_rng(seed)
            shuffled = pd.Series(rng.permutation(treated.to_numpy()))
            placebo = float(y[shuffled].mean() - y[~shuffled].mean())
            refutations = [
                {
                    "name": "placebo_shuffle",
                    "status": "pass" if abs(placebo) < abs(effect) else "warn",
                    "detail": f"placebo_effect={placebo:.6f}",
                }
            ]
            assumptions = [
                "WARNING: This is an association analysis, not causal inference.",
                "No confounding adjustment applied (dowhy unavailable or no confounders).",
                "Treatment-outcome relationship may be confounded.",
            ]

        findings = [
            {
                "kind": "recommendation",
                "treatment_column": t_col,
                "outcome_column": y_col,
                "effect_estimate": round(effect, 6),
                "method": method,
                "assumptions": assumptions,
                "confounders_adjusted": confounders if dowhy_result else [],
            }
        ]
        artifact_payload = {
            "schema_version": "causal.v2",
            "plugin_id": "causal_recommendations_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "assumptions": assumptions,
            "effect_estimate": {
                "status": "identified",
                "value": round(effect, 6),
                "method": method,
            },
            "refutations": refutations,
        }
        out_dir = ctx.artifacts_dir("causal_recommendations_v1")
        out_path = out_dir / "causal_recommendations.json"
        write_json(out_path, artifact_payload)
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="causal recommendation artifact",
            )
        ]
        return PluginResult(
            status="ok",
            summary=f"Computed causal estimate via {method}",
            metrics={
                "rows_used": int(len(t)),
                "effect_estimate": round(effect, 6),
                "method": method,
            },
            findings=findings,
            artifacts=artifacts,
            references=[],
            debug={"treatment_column": t_col, "outcome_column": y_col, "method": method},
        )
