from __future__ import annotations

import numpy as np
import pandas as pd

from statistic_harness.core.stat_plugins.surface import build_surface, top_hotspots


def test_build_surface_deterministic_grid():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "x_coord": rng.normal(size=n),
            "y_coord": rng.normal(size=n),
            "z": rng.normal(size=n),
            "extra": rng.normal(size=n),
        }
    )
    s1 = build_surface(df, z_column="z", x_column="x_coord", y_column="y_coord", numeric_columns=["z", "extra"])
    s2 = build_surface(df, z_column="z", x_column="x_coord", y_column="y_coord", numeric_columns=["z", "extra"])
    assert s1 is not None
    assert s2 is not None
    assert np.allclose(s1.z, s2.z, equal_nan=True)
    hs = top_hotspots(s1, top_n=3)
    assert len(hs) == 3
    assert all("z" in row for row in hs)

