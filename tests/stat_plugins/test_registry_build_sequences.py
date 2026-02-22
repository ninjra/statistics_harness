from __future__ import annotations

import pandas as pd

from statistic_harness.core.stat_plugins.registry import _build_sequences


def test_build_sequences_sorts_once_and_groups_by_case() -> None:
    df = pd.DataFrame(
        {
            "case_id": ["c2", "c1", "c1", "c2", "c1"],
            "event": ["E3", "E2", "E1", "E4", "E3"],
            "ts": [
                "2026-01-01T00:03:00Z",
                "2026-01-01T00:02:00Z",
                "2026-01-01T00:01:00Z",
                "2026-01-01T00:04:00Z",
                "2026-01-01T00:03:00Z",
            ],
        }
    )
    seqs = _build_sequences(df, "event", "ts", "case_id")
    assert seqs == [["E1", "E2", "E3"], ["E3", "E4"]]

