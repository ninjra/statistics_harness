"""Synthetic datasets with analytically known answers for plugin verification.

Each generator returns (pd.DataFrame, dict[str, Any]) where the dict contains
the ground-truth properties that verification checks can assert against.
All generators use a fixed seed for reproducibility.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SEED = 12345


def ds_changepoint_known() -> tuple[pd.DataFrame, dict[str, Any]]:
    """N(0,1) -> N(3,1) shift at row 200. 400 rows."""
    rng = np.random.default_rng(SEED)
    pre = rng.normal(0, 1, 200)
    post = rng.normal(3, 1, 200)
    values = np.concatenate([pre, post])
    timestamps = pd.date_range("2024-01-01", periods=400, freq="h")
    df = pd.DataFrame({"timestamp": timestamps, "value": values})
    known = {
        "changepoint_indices": [200],
        "changepoint_tolerance": 25,
        "pre_mean": 0.0,
        "post_mean": 3.0,
        "shift_magnitude": 3.0,
        "has_changepoint": True,
    }
    return df, known


def ds_no_changepoint() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pure N(0,1). 400 rows. No changepoints."""
    rng = np.random.default_rng(SEED + 1)
    values = rng.normal(0, 1, 400)
    timestamps = pd.date_range("2024-01-01", periods=400, freq="h")
    df = pd.DataFrame({"timestamp": timestamps, "value": values})
    known = {
        "has_changepoint": False,
        "expected_no_findings_at_p05": True,
    }
    return df, known


def ds_known_normal() -> tuple[pd.DataFrame, dict[str, Any]]:
    """N(0,1). 1000 rows. KS vs normal should not reject."""
    rng = np.random.default_rng(SEED + 2)
    values = rng.normal(0, 1, 1000)
    df = pd.DataFrame({"value": values})
    known = {
        "distribution": "normal",
        "mean": 0.0,
        "std": 1.0,
        "ks_vs_normal_should_not_reject": True,
    }
    return df, known


def ds_heavy_tail() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pareto(alpha=1.5). 1000 rows. Hill index ~ 1.5."""
    rng = np.random.default_rng(SEED + 3)
    alpha = 1.5
    # Pareto: x = x_m / U^(1/alpha), U ~ Uniform(0,1)
    u = rng.uniform(0, 1, 1000)
    values = 1.0 / (u ** (1.0 / alpha))
    df = pd.DataFrame({"value": values})
    known = {
        "distribution": "pareto",
        "tail_index": alpha,
        "tail_index_tolerance": 0.5,
    }
    return df, known


def ds_known_correlation() -> tuple[pd.DataFrame, dict[str, Any]]:
    """a,b correlated (r~0.99); c,d independent. 500 rows."""
    rng = np.random.default_rng(SEED + 4)
    a = rng.normal(0, 1, 500)
    b = 0.9 * a + 0.1 * rng.normal(0, 1, 500)
    c = rng.normal(0, 1, 500)
    d = rng.normal(0, 1, 500)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d})
    known = {
        "correlated_pairs": [("a", "b")],
        "independent_pairs": [("c", "d")],
        "expected_corr_ab_min": 0.85,
    }
    return df, known


def ds_known_clusters() -> tuple[pd.DataFrame, dict[str, Any]]:
    """3 Gaussian blobs, 200 each. 600 rows."""
    rng = np.random.default_rng(SEED + 5)
    centers = [(0, 0), (5, 5), (10, 0)]
    xs, ys, labels = [], [], []
    for i, (cx, cy) in enumerate(centers):
        xs.append(rng.normal(cx, 0.5, 200))
        ys.append(rng.normal(cy, 0.5, 200))
        labels.extend([i] * 200)
    df = pd.DataFrame({
        "x1": np.concatenate(xs),
        "x2": np.concatenate(ys),
        "cluster_label": labels,
    })
    known = {
        "n_clusters": 3,
        "cluster_sizes": [200, 200, 200],
    }
    return df, known


def ds_categorical_assoc() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Strong association (a,b), independent (a,c). 500 rows."""
    rng = np.random.default_rng(SEED + 6)
    cats_a = rng.choice(["X", "Y", "Z"], size=500)
    # b follows a with ~90% fidelity
    b_map = {"X": "P", "Y": "Q", "Z": "R"}
    cats_b = np.array([
        b_map[v] if rng.random() < 0.9 else rng.choice(["P", "Q", "R"])
        for v in cats_a
    ])
    # c is independent of a
    cats_c = rng.choice(["M", "N", "O"], size=500)
    df = pd.DataFrame({"a": cats_a, "b": cats_b, "c": cats_c})
    known = {
        "associated_pairs": [("a", "b")],
        "independent_pairs": [("a", "c")],
        "chi2_ab_should_reject": True,
        "chi2_ac_should_not_reject": True,
    }
    return df, known


def ds_process_log() -> tuple[pd.DataFrame, dict[str, Any]]:
    """50 cases, 5 activities, known cycle time ~10h. 300 rows."""
    rng = np.random.default_rng(SEED + 7)
    activities = ["Start", "Review", "Approve", "Process", "Close"]
    rows = []
    base = pd.Timestamp("2024-01-01")
    for case_id in range(50):
        t = base + pd.Timedelta(hours=rng.uniform(0, 200))
        n_acts = rng.integers(4, 8)  # 4-7 activities per case
        for step in range(n_acts):
            act = activities[min(step, len(activities) - 1)]
            resource = f"agent_{rng.integers(1, 6)}"
            rows.append({
                "case_id": f"case_{case_id:03d}",
                "activity": act,
                "timestamp": t,
                "resource": resource,
            })
            # Each step takes ~2h on average, total ~10h for 5 steps
            t += pd.Timedelta(hours=rng.exponential(2.0))
    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    known = {
        "n_cases": 50,
        "n_activities": 5,
        "mean_cycle_time_hours_approx": 10.0,
        "cycle_time_tolerance_hours": 5.0,
    }
    return df, known


def ds_two_pop_shift() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Group 0: N(0,1), Group 1: N(1,1). 200 each. Cohen's d = 1.0."""
    rng = np.random.default_rng(SEED + 8)
    g0 = rng.normal(0, 1, 200)
    g1 = rng.normal(1, 1, 200)
    df = pd.DataFrame({
        "group": ["control"] * 200 + ["treatment"] * 200,
        "value": np.concatenate([g0, g1]),
    })
    known = {
        "cohens_d": 1.0,
        "cohens_d_tolerance": 0.3,
        "cliffs_delta_approx": 0.76,
        "cliffs_delta_tolerance": 0.15,
        "two_sample_should_reject": True,
    }
    return df, known


def ds_determinism() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Same as ds_changepoint_known. Used for determinism testing."""
    df, known = ds_changepoint_known()
    known["purpose"] = "determinism"
    return df, known


def ds_survival() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Exponential(rate=0.1), 80% events. 300 rows. Median ~ 6.93."""
    rng = np.random.default_rng(SEED + 9)
    rate = 0.1
    durations = rng.exponential(1.0 / rate, 300)
    # 80% events, 20% censored
    events = np.ones(300, dtype=int)
    censor_idx = rng.choice(300, size=60, replace=False)
    events[censor_idx] = 0
    # Censored durations are truncated at random points
    for idx in censor_idx:
        durations[idx] *= rng.uniform(0.3, 0.9)
    groups = np.array(["A"] * 150 + ["B"] * 150)
    df = pd.DataFrame({
        "duration": durations,
        "event": events,
        "group": groups,
    })
    known = {
        "true_median_survival": np.log(2) / rate,  # ~6.93
        "median_survival_tolerance": 2.5,
        "event_rate": 0.80,
    }
    return df, known


def ds_seasonal() -> tuple[pd.DataFrame, dict[str, Any]]:
    """sin(2*pi*day/365) + noise. 730 rows (2 years). Period ~ 365."""
    rng = np.random.default_rng(SEED + 10)
    days = np.arange(730)
    signal = 3.0 * np.sin(2 * np.pi * days / 365.0)
    noise = rng.normal(0, 0.5, 730)
    values = signal + noise
    timestamps = pd.date_range("2024-01-01", periods=730, freq="D")
    df = pd.DataFrame({"timestamp": timestamps, "value": values})
    known = {
        "period_days": 365,
        "period_tolerance_days": 30,
        "amplitude": 3.0,
    }
    return df, known


def ds_benford() -> tuple[pd.DataFrame, dict[str, Any]]:
    """1000 log-uniform values + 100 anomalous (uniform 10-99). 1100 rows."""
    rng = np.random.default_rng(SEED + 11)
    # Log-uniform follows Benford's law
    log_uniform = 10 ** rng.uniform(1, 6, 1000)
    # Anomalous: uniform integers have flat leading digit distribution
    anomalous = rng.integers(10, 100, 100).astype(float)
    values = np.concatenate([log_uniform, anomalous])
    is_anomalous = np.array([0] * 1000 + [1] * 100)
    df = pd.DataFrame({"value": values, "is_anomaly": is_anomalous})
    known = {
        "n_normal": 1000,
        "n_anomalous": 100,
        "benford_should_reject_anomalous_subset": True,
    }
    return df, known


def ds_minimal() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Degenerate input. 5 rows. Plugins should skip or degrade gracefully."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": ["x", "y", "x", "y", "x"]})
    known = {
        "n_rows": 5,
        "expected_statuses": {"skipped", "degraded", "na", "ok"},
    }
    return df, known


def ds_causal_dag() -> tuple[pd.DataFrame, dict[str, Any]]:
    """X -> Y -> Z linear DAG. 300 rows."""
    rng = np.random.default_rng(SEED + 12)
    x = rng.normal(0, 1, 300)
    y = 0.7 * x + rng.normal(0, 0.3, 300)
    z = 0.5 * y + rng.normal(0, 0.3, 300)
    df = pd.DataFrame({"X": x, "Y": y, "Z": z})
    known = {
        "true_edges": [("X", "Y"), ("Y", "Z")],
        "non_edges": [("X", "Z")],
        "x_to_y_coeff": 0.7,
        "y_to_z_coeff": 0.5,
    }
    return df, known


# Lookup table for dataset generators
ALL_DATASETS: dict[str, callable] = {
    "ds_changepoint_known": ds_changepoint_known,
    "ds_no_changepoint": ds_no_changepoint,
    "ds_known_normal": ds_known_normal,
    "ds_heavy_tail": ds_heavy_tail,
    "ds_known_correlation": ds_known_correlation,
    "ds_known_clusters": ds_known_clusters,
    "ds_categorical_assoc": ds_categorical_assoc,
    "ds_process_log": ds_process_log,
    "ds_two_pop_shift": ds_two_pop_shift,
    "ds_determinism": ds_determinism,
    "ds_survival": ds_survival,
    "ds_seasonal": ds_seasonal,
    "ds_benford": ds_benford,
    "ds_minimal": ds_minimal,
    "ds_causal_dag": ds_causal_dag,
}
