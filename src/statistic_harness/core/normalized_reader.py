from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from .types import PluginContext


@dataclass(frozen=True)
class ReadStats:
    rows: int = 0
    batches: int = 0
    used_iter_batches: bool = False
    used_loader: bool = False


def iter_normalized_batches(
    ctx: PluginContext,
    *,
    columns: list[str] | None = None,
    batch_size: int | None = None,
    row_limit: int | None = None,
) -> Iterable[pd.DataFrame]:
    """Canonical streaming read path for normalized-layer data."""

    if callable(ctx.dataset_iter_batches):
        return ctx.dataset_iter_batches(
            columns=columns,
            batch_size=batch_size,
            row_limit=row_limit,
        )
    accessor = ctx.dataset_loader()
    if hasattr(accessor, "iter_batches"):
        return accessor.iter_batches(
            columns=columns,
            batch_size=int(batch_size or 100_000),
            row_limit=row_limit,
        )
    raise RuntimeError("dataset_iter_batches unavailable and accessor has no iter_batches")


def load_normalized_df(
    ctx: PluginContext,
    *,
    columns: list[str] | None = None,
    row_limit: int | None = None,
) -> pd.DataFrame:
    """Fallback bounded dataframe load for non-streaming plugin logic."""

    return ctx.dataset_loader(columns=columns, row_limit=row_limit)


def read_with_stats(
    ctx: PluginContext,
    *,
    columns: list[str] | None = None,
    batch_size: int | None = None,
    row_limit: int | None = None,
) -> tuple[list[pd.DataFrame], ReadStats]:
    batches: list[pd.DataFrame] = []
    rows = 0
    for batch in iter_normalized_batches(
        ctx,
        columns=columns,
        batch_size=batch_size,
        row_limit=row_limit,
    ):
        batches.append(batch)
        rows += int(len(getattr(batch, "index", [])))
    return batches, ReadStats(
        rows=rows,
        batches=len(batches),
        used_iter_batches=True,
        used_loader=False,
    )

