from __future__ import annotations

from pathlib import Path

import pandas as pd

from statistic_harness.core.normalized_reader import iter_normalized_batches, read_with_stats
from statistic_harness.core.types import PluginContext


def _fake_batches():
    for idx in range(2):
        yield pd.DataFrame({"x": [idx, idx + 1]})


def test_iter_normalized_batches_uses_ctx_iterator() -> None:
    ctx = PluginContext(
        run_id="r1",
        run_dir=Path("."),
        settings={},
        run_seed=1,
        logger=lambda _: None,
        storage=None,
        dataset_loader=lambda **_: pd.DataFrame(),
        dataset_iter_batches=lambda **_: _fake_batches(),
    )
    out = list(iter_normalized_batches(ctx))
    assert len(out) == 2


def test_read_with_stats_counts_rows() -> None:
    ctx = PluginContext(
        run_id="r1",
        run_dir=Path("."),
        settings={},
        run_seed=1,
        logger=lambda _: None,
        storage=None,
        dataset_loader=lambda **_: pd.DataFrame(),
        dataset_iter_batches=lambda **_: _fake_batches(),
    )
    batches, stats = read_with_stats(ctx)
    assert len(batches) == 2
    assert stats.rows == 4
    assert stats.batches == 2

