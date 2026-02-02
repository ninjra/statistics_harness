from __future__ import annotations

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# Linear data
n = 200
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
y = 2 * x1 - 0.5 * x2 + rng.normal(scale=0.1, size=n)
# Inject outliers
outlier_idx = [10, 50, 150]
y[outlier_idx] += 5
linear = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
linear.to_csv("tests/fixtures/synth_linear.csv", index=False)

# Timeseries with mean shift
values = np.concatenate(
    [rng.normal(loc=0, scale=1, size=100), rng.normal(loc=3, scale=1, size=100)]
)
timeseries = pd.DataFrame({"value": values})
timeseries.to_csv("tests/fixtures/synth_timeseries.csv", index=False)

# Clusters
cluster1 = rng.normal(loc=0, scale=0.5, size=(50, 2))
cluster2 = rng.normal(loc=3, scale=0.5, size=(50, 2))
clusters = pd.DataFrame(np.vstack([cluster1, cluster2]), columns=["x", "y"])
clusters.to_csv("tests/fixtures/synth_clusters.csv", index=False)

# Shifted correlation
a = rng.normal(size=100)
b = rng.normal(size=100)
first = pd.DataFrame({"a": a, "b": b})
second = pd.DataFrame(
    {"a": rng.normal(size=100), "b": a + rng.normal(scale=0.1, size=100)}
)
shift = pd.concat([first, second], ignore_index=True)
shift.to_csv("tests/fixtures/synth_shift_corr.csv", index=False)
