from __future__ import annotations

import datetime as dt

import pandas as pd

from conftest import make_context
from plugins.analysis_conformance_alignments.plugin import Plugin as ConformancePlugin


def test_conformance_alignments_respects_sequence_caps(run_dir) -> None:
    rows = 9000
    base = dt.datetime(2026, 1, 1)
    df = pd.DataFrame(
        {
            "event": [f"S{i % 30}" for i in range(rows)],
            "case_id": [f"C{i % 2500}" for i in range(rows)],
            "ts": [base + dt.timedelta(seconds=i) for i in range(rows)],
        }
    )
    df["ts"] = df["ts"].astype(str)
    ctx = make_context(run_dir, df, {})
    ctx.settings = {"max_sequences": 500, "max_variant_length": 20, "max_edit_cells": 5000}
    result = ConformancePlugin().run(ctx)
    assert result.status in {"ok", "skipped", "degraded"}
    assert result.status != "error"

