from __future__ import annotations


import numpy as np
from sklearn.linear_model import Ridge

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


class Plugin:
    def run(self, ctx) -> PluginResult:
        df = ctx.dataset_loader()
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            return PluginResult("skipped", "No numeric columns", {}, [], [], None)

        max_cols = int(ctx.settings.get("max_target_cols", 5))
        alpha = float(ctx.settings.get("alpha", 0.1))
        findings = []
        artifacts = []
        artifacts_dir = ctx.artifacts_dir("analysis_conformal_feature_prediction")

        for target in list(numeric.columns)[:max_cols]:
            y = numeric[target].to_numpy()
            X = numeric.drop(columns=[target]).to_numpy()
            if X.size == 0:
                continue
            n = len(y)
            n_train = int(0.6 * n)
            n_calib = int(0.2 * n)
            train_idx = slice(0, n_train)
            calib_idx = slice(n_train, n_train + n_calib)
            test_idx = slice(n_train + n_calib, n)

            model = Ridge(alpha=1.0)
            model.fit(X[train_idx], y[train_idx])
            calib_pred = model.predict(X[calib_idx])
            resid = np.abs(y[calib_idx] - calib_pred)
            q = np.quantile(resid, 1 - alpha)
            test_pred = model.predict(X[test_idx])
            lower = test_pred - q
            upper = test_pred + q
            test_y = y[test_idx]
            anomaly_mask = (test_y < lower) | (test_y > upper)
            test_indices = np.arange(n)[test_idx]

            for idx, is_anom, lo, hi, score in zip(
                test_indices, anomaly_mask, lower, upper, np.abs(test_y - test_pred)
            ):
                if is_anom:
                    findings.append(
                        {
                            "kind": "anomaly",
                            "column": target,
                            "row_index": int(idx),
                            "score": float(score),
                            "lower": float(lo),
                            "upper": float(hi),
                        }
                    )

        anomalies_path = artifacts_dir / "anomalies.json"
        write_json(anomalies_path, findings)
        artifacts.append(
            PluginArtifact(
                path=str(anomalies_path.relative_to(ctx.run_dir)),
                type="json",
                description="Anomalies",
            )
        )

        if not findings:
            return PluginResult(
                "skipped",
                "No anomalies detected",
                {"count": 0},
                [],
                artifacts,
                None,
            )

        return PluginResult(
            "ok",
            "Computed conformal anomalies",
            {"count": len(findings)},
            findings,
            artifacts,
            None,
        )
