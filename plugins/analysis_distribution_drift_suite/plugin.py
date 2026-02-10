from __future__ import annotations

from typing import Any

import math

import numpy as np
import pandas as pd

from statistic_harness.core.stat_plugins import (
    BudgetTimer,
    merge_config,
    infer_columns,
    deterministic_sample,
    bh_fdr,
    standardized_median_diff,
    cliffs_delta,
    cramers_v,
    stable_id,
)
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


DEFAULTS = {
    "drift": {
        "alpha": 0.05,
        "window_strategy": "early_vs_late",
        "rolling_window_count": 6,
        "min_group_size": 100,
        "mmd": {"enabled": False, "max_points": 5000, "permutations": 200},
    }
}


def _ks_stat(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.sort(x)
    y = np.sort(y)
    n1 = x.size
    n2 = y.size
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    data_all = np.concatenate([x, y])
    cdf1 = np.searchsorted(x, data_all, side="right") / n1
    cdf2 = np.searchsorted(y, data_all, side="right") / n2
    d = np.max(np.abs(cdf1 - cdf2))
    n_eff = n1 * n2 / (n1 + n2)
    p = min(1.0, 2.0 * math.exp(-2.0 * (d**2) * n_eff))
    return float(d), float(p)


def _chi2_p_value(chi2: float, df: int) -> float:
    try:
        from scipy.stats import chi2 as chi2_dist  # type: ignore

        return float(chi2_dist.sf(chi2, df))
    except Exception:
        return float(math.exp(-0.5 * chi2))


def _group_slices(df: pd.DataFrame, group_cols: list[str], max_groups: int) -> list[tuple[str, pd.DataFrame]]:
    slices: list[tuple[str, pd.DataFrame]] = [("ALL", df)]
    for col in group_cols:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=False)
        for value in counts.index[:max_groups]:
            slices.append((f"{col}={value}", df.loc[df[col] == value]))
    return slices


class Plugin:
    def run(self, ctx) -> PluginResult:
        config = merge_config(ctx.settings)
        config["drift"] = {**DEFAULTS["drift"], **config.get("drift", {})}
        timer = BudgetTimer(config.get("time_budget_ms"))

        df = ctx.dataset_loader()
        if df.empty:
            return PluginResult("skipped", "Empty dataset", {}, [], [], None)

        df, sample_meta = deterministic_sample(df, config.get("max_rows"), seed=config.get("seed", 1337))
        inferred = infer_columns(df, config)
        value_cols = inferred.get("value_columns") or []
        cat_cols = inferred.get("categorical_columns") or []
        time_col = inferred.get("time_column")
        group_cols = inferred.get("group_by") or []

        if not value_cols and not cat_cols:
            return PluginResult("skipped", "No analyzable columns detected", {}, [], [], None)

        if time_col and time_col in df.columns:
            df = df.sort_values(time_col)

        drift_cfg = config["drift"]
        alpha = float(drift_cfg.get("alpha", 0.05))
        min_group = int(drift_cfg.get("min_group_size", 200))
        max_findings = int(config.get("max_findings", 30))
        max_groups = int(config.get("max_groups", 30))

        comparisons: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
        if time_col and time_col in df.columns:
            n = df.shape[0]
            cut = int(n * 0.3)
            if cut > 0:
                comparisons.append(("early_vs_late", df.iloc[:cut], df.iloc[-cut:]))
        focus_windows = config.get("focus", {}).get("windows") if isinstance(config.get("focus"), dict) else []
        if focus_windows and time_col and time_col in df.columns:
            ts = pd.to_datetime(df[time_col], errors="coerce")
            for window in focus_windows:
                if not isinstance(window, dict):
                    continue
                name = window.get("name") or "window"
                start = pd.to_datetime(window.get("start"), errors="coerce")
                end = pd.to_datetime(window.get("end"), errors="coerce")
                if pd.isna(start) or pd.isna(end):
                    continue
                mask = (ts >= start) & (ts <= end)
                comparisons.append((name, df.loc[mask], df.loc[~mask]))

        if not comparisons:
            return PluginResult("skipped", "No comparison windows available", {}, [], [], None)

        tests: list[dict[str, Any]] = []
        for label, slice_a, slice_b in comparisons:
            if timer.exceeded():
                break
            for group_label, group_df in _group_slices(df, group_cols, max_groups):
                if timer.exceeded():
                    break
                if group_label != "ALL":
                    group_df = group_df.copy()
                    slice_a = slice_a.loc[group_df.index.intersection(slice_a.index)]
                    slice_b = slice_b.loc[group_df.index.intersection(slice_b.index)]
                if slice_a.shape[0] < min_group or slice_b.shape[0] < min_group:
                    continue
                for col in value_cols:
                    if timer.exceeded():
                        break
                    x = slice_a[col].dropna().to_numpy(dtype=float)
                    y = slice_b[col].dropna().to_numpy(dtype=float)
                    if x.size < min_group or y.size < min_group:
                        continue
                    d, p = _ks_stat(x, y)
                    effect = standardized_median_diff(x, y)
                    tests.append(
                        {
                            "kind": "numeric_drift",
                            "column": col,
                            "group": group_label,
                            "window": label,
                            "stat": d,
                            "p_value": p,
                            "effect": effect,
                            "direction": "higher" if np.median(y) > np.median(x) else "lower",
                        }
                    )
                for col in cat_cols:
                    if timer.exceeded():
                        break
                    a_counts = slice_a[col].fillna("NA").astype(str).value_counts()
                    b_counts = slice_b[col].fillna("NA").astype(str).value_counts()
                    categories = list(set(a_counts.index).union(set(b_counts.index)))
                    if len(categories) < 2:
                        continue
                    obs = np.array(
                        [
                            [a_counts.get(cat, 0) for cat in categories],
                            [b_counts.get(cat, 0) for cat in categories],
                        ],
                        dtype=float,
                    )
                    total = obs.sum()
                    if total <= 0:
                        continue
                    row_sums = obs.sum(axis=1, keepdims=True)
                    col_sums = obs.sum(axis=0, keepdims=True)
                    expected = row_sums @ col_sums / total
                    with np.errstate(divide="ignore", invalid="ignore"):
                        chi2 = np.nansum((obs - expected) ** 2 / expected)
                    df_chi = (obs.shape[0] - 1) * (obs.shape[1] - 1)
                    p = _chi2_p_value(float(chi2), int(df_chi))
                    effect = cramers_v(obs)
                    tests.append(
                        {
                            "kind": "categorical_drift",
                            "column": col,
                            "group": group_label,
                            "window": label,
                            "stat": float(chi2),
                            "p_value": p,
                            "effect": effect,
                            "direction": "shift",
                        }
                    )

        p_values = [t["p_value"] for t in tests]
        q_values, _ = bh_fdr(p_values, alpha)
        findings: list[dict[str, Any]] = []
        for test, q in zip(tests, q_values):
            if q > alpha:
                continue
            findings.append(
                {
                    "id": stable_id(f"{test['kind']}:{test['column']}:{test['group']}:{test['window']}"),
                    "severity": "warn",
                    "confidence": max(0.2, 1.0 - float(q)),
                    "title": f"Drift detected in {test['column']} ({test['group']})",
                    "what": f"{test['kind']} changed in {test['window']} comparison.",
                    "why": f"p={test['p_value']:.3g}, q={q:.3g}, effect={test['effect']:.3g}.",
                    "evidence": {
                        "metrics": {
                            "stat": test["stat"],
                            "p_value": test["p_value"],
                            "q_value": q,
                            "effect": test["effect"],
                            "direction": test["direction"],
                        }
                    },
                    "where": {
                        "column": test["column"],
                        "group": test["group"],
                        "window": test["window"],
                    },
                    "recommendation": "Investigate root causes for distribution shift.",
                    "measurement_type": "measured",
                    "references": [
                        {
                            "title": "Benjamini & Hochberg (1995) FDR",
                            "url": "https://doi.org/10.1111/j.2517-6161.1995.tb02031.x",
                            "doi": "10.1111/j.2517-6161.1995.tb02031.x",
                        }
                    ],
                }
            )
            if len(findings) >= max_findings:
                break

        artifacts_dir = ctx.artifacts_dir("analysis_distribution_drift_suite")
        out_path = artifacts_dir / "drift_table.json"
        write_json(out_path, {"tests": tests, "q_values": q_values})
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Distribution drift tests",
            )
        ]

        metrics = {
            "rows_seen": int(sample_meta.get("rows_total", len(df))),
            "rows_used": int(sample_meta.get("rows_used", len(df))),
            "cols_used": len(value_cols) + len(cat_cols),
            "references": [
                {
                    "title": "Benjamini & Hochberg (1995) FDR",
                    "url": "https://doi.org/10.1111/j.2517-6161.1995.tb02031.x",
                    "doi": "10.1111/j.2517-6161.1995.tb02031.x",
                }
            ],
        }

        summary = f"Detected {len(findings)} drift findings." if findings else "No significant drift detected."

        return PluginResult(
            "ok",
            summary,
            metrics,
            findings,
            artifacts,
            None,
        )
