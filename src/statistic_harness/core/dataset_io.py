from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class DatasetAccessor:
    def __init__(self, canonical_path: Path) -> None:
        self.canonical_path = canonical_path
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self.canonical_path)
        return self._df.copy()

    def info(self) -> dict[str, Any]:
        df = self.load()
        inferred = {col: str(dtype) for col, dtype in df.dtypes.items()}
        return {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "inferred_types": inferred,
        }
