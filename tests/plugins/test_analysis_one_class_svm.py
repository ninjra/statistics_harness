import datetime as dt

import numpy as np
import pandas as pd

from plugins.analysis_one_class_svm.plugin import Plugin
from tests.conftest import make_context


def _sample_df() -> pd.DataFrame:
    rows = 200
    return pd.DataFrame(
        {
            "metric": [0.1] * 100 + [2.0] * 100,
            "metric2": [1.0] * 50 + [3.0] * 150,
            "category": ["A"] * 100 + ["B"] * 100,
            "text": ["Error code 500"] * 100 + ["Timeout at step 3"] * 100,
            "case_id": [i // 4 for i in range(rows)],
            "ts": [
                dt.datetime(2026, 1, 1) + dt.timedelta(minutes=i)
                for i in range(rows)
            ],
        }
    )


def test_analysis_one_class_svm_smoke(run_dir):
    df = _sample_df()
    df["ts"] = df["ts"].astype(str)
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status in ("ok", "skipped")


def test_analysis_one_class_svm_large_n_uses_deterministic_fallback(run_dir):
    rows = 60000
    rng = np.random.default_rng(1337)
    df = pd.DataFrame(
        {
            "metric": rng.normal(0.0, 1.0, rows),
            "metric2": rng.normal(1.0, 2.0, rows),
        }
    )
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    debug = result.debug if isinstance(result.debug, dict) else {}
    assert debug.get("model_path") == "robust_z_fallback_large_n"
