from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

from statistic_harness.core.storage import Storage
from statistic_harness.core.types import PluginContext


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "dataset").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir


def make_context(run_dir: Path, df: pd.DataFrame, settings: dict) -> PluginContext:
    canonical = run_dir / "dataset" / "canonical.csv"
    df.to_csv(canonical, index=False)
    storage = Storage(run_dir / "state.sqlite")

    def loader() -> pd.DataFrame:
        return pd.read_csv(canonical)

    def logger(msg: str) -> None:
        pass

    return PluginContext(
        run_id="test-run",
        run_dir=run_dir,
        settings=settings,
        run_seed=42,
        logger=logger,
        storage=storage,
        dataset_loader=loader,
    )
