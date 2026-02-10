from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from statistic_harness.core.stat_plugins import (
    BudgetTimer,
    merge_config,
    infer_columns,
    deterministic_sample,
    robust_center_scale,
    stable_id,
)
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


DEFAULTS = {
    "control_chart": {
        "z_thresh": 4.0,
        "ewma_lambda": 0.2,
        "ewma_L": 3.0,
        "cusum_k": 0.5,
        "cusum_h": 5.0,
        "min_points": 50,
        "handle_autocorr": "off",
        "max_series_points": 5000,
    }
}


def _downsample_series(values: list[float], limit: int) -> list[float]:
    if limit <= 0 or len(values) <= limit:
        return values
    idx = np.linspace(0, len(values) - 1, num=limit, dtype=int)
    return [values[i] for i in idx]


def _ewma(series: np.ndarray, lam: float) -> np.ndarray:
    ewma = np.zeros_like(series, dtype=float)
    if series.size == 0:
        return ewma
    ewma[0] = series[0]
    for idx in range(1, series.size):
        ewma[idx] = lam * series[idx] + (1.0 - lam) * ewma[idx - 1]
    return ewma


def _cusum(series: np.ndarray, center: float, k: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    c_plus = np.zeros_like(series, dtype=float)
    c_minus = np.zeros_like(series, dtype=float)
    for idx, value in enumerate(series):
        if idx == 0:
            c_plus[idx] = max(0.0, value - (center + k))
            c_minus[idx] = max(0.0, (center - k) - value)
        else:
            c_plus[idx] = max(0.0, c_plus[idx - 1] + value - (center + k))
            c_minus[idx] = max(0.0, c_minus[idx - 1] + (center - k) - value)
    return c_plus, c_minus


def _group_slices(df: pd.DataFrame, group_cols: list[str], max_groups: int) -> list[tuple[str, pd.DataFrame]]:
    slices: list[tuple[str, pd.DataFrame]] = [("ALL", df)]
    for col in group_cols:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=False)
        for value in counts.index[:max_groups]:
            label = f"{col}={value}"
            slices.append((label, df.loc[df[col] == value]))
    return slices


class Plugin:
    def run(self, ctx) -> PluginResult:
        config = merge_config(ctx.settings)
        config["control_chart"] = {**DEFAULTS["control_chart"], **config.get("control_chart", {})}
        timer = BudgetTimer(config.get("time_budget_ms"))

        df = ctx.dataset_loader()
        if df.empty:
            return PluginResult("skipped", "Empty dataset", {}, [], [], None)

        df, sample_meta = deterministic_sample(df, config.get("max_rows"), seed=config.get("seed", 1337))
        inferred = infer_columns(df, config)
        value_cols = inferred.get("value_columns") or []
        time_col = inferred.get("time_column")
        group_cols = inferred.get("group_by") or []

        if not value_cols:
            return PluginResult("skipped", "No numeric columns detected", {}, [], [], None)

        if time_col and time_col in df.columns:
            df = df.sort_values(time_col)

        settings = config["control_chart"]
        min_points = int(settings.get("min_points", 50))
        max_findings = int(config.get("max_findings", 30))
        max_groups = int(config.get("max_groups", 30))

        findings: list[dict[str, Any]] = []
        alarms: list[dict[str, Any]] = []

        for value_col in value_cols:
            if timer.exceeded():
                break
            slices = _group_slices(df, group_cols, max_groups)
            for label, slice_df in slices:
                if timer.exceeded():
                    break
                series = slice_df[value_col].dropna()
                if series.shape[0] < min_points:
                    continue
                values = series.to_numpy(dtype=float)
                center, scale = robust_center_scale(values)
                if scale <= 0 or not np.isfinite(scale):
                    continue
                zscores = (values - center) / scale
                z_thresh = float(settings.get("z_thresh", 4.0))
                indiv_mask = np.abs(zscores) > z_thresh

                lam = float(settings.get("ewma_lambda", 0.2))
                ewma = _ewma(values, lam)
                sigma_ewma = scale * np.sqrt(lam / (2.0 - lam))
                ewma_L = float(settings.get("ewma_L", 3.0))
                ewma_mask = np.abs(ewma - center) > (ewma_L * sigma_ewma)

                k = float(settings.get("cusum_k", 0.5)) * scale
                h = float(settings.get("cusum_h", 5.0)) * scale
                c_plus, c_minus = _cusum(values, center, k, h)
                cusum_mask = (c_plus > h) | (c_minus > h)

                if not (indiv_mask.any() or ewma_mask.any() or cusum_mask.any()):
                    continue

                last_idx = int(np.max(np.where(indiv_mask | ewma_mask | cusum_mask)))
                last_value = float(values[last_idx])
                direction = "high" if last_value >= center else "low"
                run_length = 0
                for flag in reversed(indiv_mask.tolist()):
                    if flag:
                        run_length += 1
                    else:
                        break

                severity = "warn"
                if (indiv_mask.sum() + ewma_mask.sum() + cusum_mask.sum()) > 3:
                    severity = "critical"

                finding = {
                    "id": stable_id(f"{value_col}:{label}:{last_idx}"),
                    "severity": severity,
                    "confidence": min(1.0, max(0.2, float(abs(zscores[last_idx]) / z_thresh))),
                    "title": f"Shift detected in {value_col} ({label})",
                    "what": f"Recent values are {direction} relative to robust baseline.",
                    "why": "Individuals/EWMA/CUSUM alarms exceeded robust thresholds.",
                    "evidence": {
                        "metrics": {
                            "center": float(center),
                            "scale": float(scale),
                            "last_value": last_value,
                            "last_index": last_idx,
                            "run_length": run_length,
                            "indiv_alarms": int(indiv_mask.sum()),
                            "ewma_alarms": int(ewma_mask.sum()),
                            "cusum_alarms": int(cusum_mask.sum()),
                        }
                    },
                    "where": {
                        "column": value_col,
                        "group": label,
                        "window": None,
                    },
                    "recommendation": "Investigate recent shift drivers and validate upstream changes.",
                    "measurement_type": "measured",
                    "references": [
                        {
                            "title": "NIST/SEMATECH e-Handbook of Statistical Methods (Control Charts)",
                            "url": "https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc3.htm",
                            "doi": "",
                        }
                    ],
                }
                findings.append(finding)
                alarms.append(
                    {
                        "column": value_col,
                        "group": label,
                        "last_index": last_idx,
                        "direction": direction,
                        "indiv_alarms": int(indiv_mask.sum()),
                        "ewma_alarms": int(ewma_mask.sum()),
                        "cusum_alarms": int(cusum_mask.sum()),
                    }
                )
                if len(findings) >= max_findings:
                    break
            if len(findings) >= max_findings:
                break

        max_points = int(settings.get("max_series_points", 5000))
        series_preview = {}
        if value_cols:
            sample_col = value_cols[0]
            series = df[sample_col].dropna().to_numpy(dtype=float).tolist()
            series_preview = {
                "column": sample_col,
                "values": _downsample_series(series, max_points),
            }

        artifacts_dir = ctx.artifacts_dir("analysis_control_chart_suite")
        out_path = artifacts_dir / "alarms.json"
        write_json(out_path, {"alarms": alarms, "series_preview": series_preview})
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Control chart alarms",
            )
        ]

        metrics = {
            "rows_seen": int(sample_meta.get("rows_total", len(df))),
            "rows_used": int(sample_meta.get("rows_used", len(df))),
            "cols_used": len(value_cols),
            "references": [
                {
                    "title": "NIST/SEMATECH e-Handbook of Statistical Methods (Control Charts)",
                    "url": "https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc3.htm",
                    "doi": "",
                }
            ],
        }

        summary = f"Detected {len(findings)} control-chart alarms." if findings else "No control-chart alarms detected."

        return PluginResult(
            "ok",
            summary,
            metrics,
            findings,
            artifacts,
            None,
        )
