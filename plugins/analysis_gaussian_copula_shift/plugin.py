from __future__ import annotations

import numpy as np

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


class Plugin:
    def run(self, ctx) -> PluginResult:
        df = ctx.dataset_loader()
        numeric = df.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            return PluginResult("skipped", "Not enough numeric columns", {}, [], [], None)
        values = numeric.to_numpy()
        n = values.shape[0]
        split = n // 2
        first = values[:split]
        second = values[split:]

        def inv_norm(p: np.ndarray) -> np.ndarray:
            # Approximation by Peter John Acklam, implemented for array.
            a = [
                -3.969683028665376e01,
                2.209460984245205e02,
                -2.759285104469687e02,
                1.383577518672690e02,
                -3.066479806614716e01,
                2.506628277459239e00,
            ]
            b = [
                -5.447609879822406e01,
                1.615858368580409e02,
                -1.556989798598866e02,
                6.680131188771972e01,
                -1.328068155288572e01,
            ]
            c = [
                -7.784894002430293e-03,
                -3.223964580411365e-01,
                -2.400758277161838e00,
                -2.549732539343734e00,
                4.374664141464968e00,
                2.938163982698783e00,
            ]
            d = [
                7.784695709041462e-03,
                3.224671290700398e-01,
                2.445134137142996e00,
                3.754408661907416e00,
            ]
            plow = 0.02425
            phigh = 1 - plow
            x = np.zeros_like(p)
            lower = p < plow
            upper = p > phigh
            mid = (~lower) & (~upper)
            q = np.sqrt(-2 * np.log(p[lower]))
            x[lower] = (
                (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
            )
            q = p[mid] - 0.5
            r = q * q
            x[mid] = (
                (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
                * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
            )
            q = np.sqrt(-2 * np.log(1 - p[upper]))
            x[upper] = -(
                (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
            )
            return x

        def to_gaussian(data: np.ndarray) -> np.ndarray:
            ranks = np.argsort(np.argsort(data, axis=0), axis=0) + 1
            u = ranks / (data.shape[0] + 1)
            u = np.clip(u, 1e-6, 1 - 1e-6)
            return inv_norm(u)

        z1 = to_gaussian(first)
        z2 = to_gaussian(second)
        corr1 = np.corrcoef(z1, rowvar=False)
        corr2 = np.corrcoef(z2, rowvar=False)
        delta = np.abs(corr1 - corr2)

        pairs = []
        for i in range(delta.shape[0]):
            for j in range(i + 1, delta.shape[1]):
                pairs.append((i, j, (numeric.columns[i], numeric.columns[j]), float(delta[i, j])))
        pairs.sort(key=lambda x: x[3], reverse=True)
        max_pairs = int(ctx.settings.get("max_pairs", 5))
        selected = pairs[:max_pairs]

        rng = np.random.default_rng(ctx.run_seed)
        findings = []
        for i, j, (a, b), score in selected:
            p_value = 1.0
            perm_scores = []
            for _ in range(int(ctx.settings.get("n_permutations", 100))):
                perm = rng.permutation(values)
                perm_first = perm[:split]
                perm_second = perm[split:]
                z1p = to_gaussian(perm_first)
                z2p = to_gaussian(perm_second)
                delta_p = np.abs(np.corrcoef(z1p, rowvar=False) - np.corrcoef(z2p, rowvar=False))
                perm_scores.append(delta_p[i, j])
            if perm_scores:
                p_value = float((np.array(perm_scores) >= score).mean())
            findings.append({"kind": "dependence_shift", "pair": [a, b], "delta": score, "p_value": p_value})

        artifacts_dir = ctx.artifacts_dir("analysis_gaussian_copula_shift")
        out_path = artifacts_dir / "summary.json"
        write_json(out_path, findings)
        artifacts = [
            PluginArtifact(path=str(out_path.relative_to(ctx.run_dir)), type="json", description="Dependence shift")
        ]
        return PluginResult("ok", "Computed copula shift", {"pairs": len(findings)}, findings, artifacts, None)
