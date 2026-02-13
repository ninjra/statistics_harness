from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from statistic_harness.core.utils import stable_hash


def deterministic_sample(
    df: pd.DataFrame,
    max_rows: int | None,
    seed: int = 1337,
    key_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if max_rows is None or len(df) <= max_rows:
        return df, {
            "rows_total": int(len(df)),
            "rows_used": int(len(df)),
            "sampled": False,
            "hash_columns": key_columns or [],
        }
    cols = _select_hash_columns(df, key_columns)
    hashes = _row_hashes(df, cols, seed)
    order = np.argsort(hashes.values)
    keep = order[: int(max_rows)]
    sampled = df.iloc[keep].copy()
    return sampled, {
        "rows_total": int(len(df)),
        "rows_used": int(len(sampled)),
        "sampled": True,
        "hash_columns": cols,
    }


def _select_hash_columns(df: pd.DataFrame, key_columns: list[str] | None) -> list[str]:
    if key_columns:
        return [col for col in key_columns if col in df.columns]
    return list(df.columns[: min(8, len(df.columns))])


def _row_hashes(df: pd.DataFrame, columns: list[str], seed: int) -> pd.Series:
    if not columns:
        base = pd.Series(range(len(df)), index=df.index, dtype="uint64")
    else:
        frame = df[columns].astype(str)
        base = pd.util.hash_pandas_object(frame, index=False, hash_key="stat_harness_key")
    if seed:
        salt = np.uint64(stable_hash(f"stat_seed:{seed}"))
        base = base.astype("uint64") ^ salt
    return base
