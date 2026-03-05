from __future__ import annotations
import logging
import traceback

import numpy as np
import pandas as pd
from statistic_harness.core.types import PluginResult, PluginArtifact, PluginError

logger = logging.getLogger(__name__)


class Plugin:
    def run(self, ctx) -> PluginResult:
        try:
            df = ctx.dataset_loader()
            if df.empty:
                return PluginResult("skipped", "Empty dataset", {}, [], [], None)

            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if len(numeric_cols) < 1:
                return PluginResult("skipped", "No numeric columns found", {}, [], [], None)

            settings = ctx.settings or {}
            state_cols = settings.get("state_columns", numeric_cols)
            # Filter to columns that actually exist
            state_cols = [c for c in state_cols if c in df.columns]
            if len(state_cols) < 1:
                return PluginResult("skipped", "No valid state columns", {}, [], [], None)

            work = df[state_cols].dropna()
            if len(work) < 10:
                return PluginResult(
                    "skipped", f"Insufficient rows ({len(work)}) for SINDy", {}, [], [], None,
                )

            try:
                import pysindy as ps
            except ImportError:
                return PluginResult("skipped", "pysindy not installed", {}, [], [], None)

            X = work.values.astype(float)
            dt = settings.get("dt", 1.0)

            # Build and fit SINDy model
            feature_names = list(state_cols)
            optimizer = ps.STLSQ(threshold=settings.get("threshold", 0.1))
            feature_lib = ps.PolynomialLibrary(degree=settings.get("poly_degree", 2))

            model = ps.SINDy(
                optimizer=optimizer,
                feature_library=feature_lib,
            )
            model.fit(X, t=dt, feature_names=feature_names)

            # Extract equations
            equations = model.equations()
            coef_matrix = model.coefficients()

            # Build coefficient map
            lib_names = model.get_feature_names()
            coef_details = {}
            for i, state_var in enumerate(feature_names):
                terms = {}
                for j, lib_name in enumerate(lib_names):
                    val = float(coef_matrix[i, j])
                    if abs(val) > 1e-12:
                        terms[lib_name] = round(val, 8)
                coef_details[state_var] = terms

            # Model score (R^2 on numerical derivatives)
            try:
                score = float(model.score(X, t=dt))
            except Exception:
                score = None

            findings = [{
                "kind": "time_series",
                "measurement_type": "measured",
                "state_columns": feature_names,
                "equations": equations,
                "coefficients": coef_details,
                "r_squared": round(score, 6) if score is not None else None,
                "n_library_terms": len(lib_names),
                "method": "SINDy (PySINDy)",
            }]

            eq_summary = "; ".join(
                f"d{state_cols[i]}/dt = {eq}" for i, eq in enumerate(equations)
            )

            return PluginResult(
                status="ok",
                summary=f"SINDy discovered {len(equations)} equations: {eq_summary}",
                metrics={
                    "n_observations": len(work),
                    "n_state_variables": len(feature_names),
                    "n_equations": len(equations),
                    "r_squared": round(score, 6) if score is not None else None,
                    "n_library_terms": len(lib_names),
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(f"SINDy dynamics discovery failed: {e}", exc_info=True)
            return PluginResult(
                status="error",
                summary=f"SINDy dynamics discovery failed: {e}",
                metrics={},
                findings=[],
                artifacts=[],
                error=PluginError(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()),
            )
