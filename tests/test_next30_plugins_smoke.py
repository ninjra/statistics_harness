from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from statistic_harness.core.stat_plugins.next30_addon import (
    HANDLERS as NEXT30_HANDLERS,
    NEXT30_IDS,
)
from tests.conftest import make_context


def _next30_dataset(rows: int = 480) -> pd.DataFrame:
    rng = np.random.default_rng(20260213)
    base = datetime(2025, 1, 1, 0, 0, 0)
    idx = np.arange(rows)

    irregular_minutes = rng.integers(0, 35, size=rows, dtype=np.int64)
    ts = [base + timedelta(hours=int(i), minutes=int(irregular_minutes[i])) for i in idx]
    shift = (idx >= (rows // 2)).astype(float)
    seasonal = np.sin((2.0 * np.pi * idx) / 24.0)
    noise = rng.normal(0.0, 0.45, size=rows)

    queue_wait = 20.0 + 4.0 * seasonal + 6.0 * shift + noise
    duration = 36.0 + 2.3 * queue_wait + 3.0 * shift + rng.normal(0.0, 1.5, size=rows)
    service_runtime = 24.0 + 1.5 * queue_wait + rng.normal(0.0, 1.2, size=rows)
    eligible_jobs = 90.0 + 0.6 * queue_wait + rng.normal(0.0, 2.0, size=rows)
    backlog = 40.0 + 0.4 * duration + rng.normal(0.0, 2.0, size=rows)
    metric_x = 10.0 + 0.8 * seasonal + rng.normal(0.0, 0.6, size=rows)
    metric_y = 7.0 + 0.5 * metric_x + rng.normal(0.0, 0.4, size=rows)

    raw_count = rng.poisson(np.where(shift > 0.5, 6.5, 3.2), size=rows)
    structural_zero = rng.random(rows) < 0.34
    event_count = np.where(structural_zero, 0, raw_count)
    retry_count = np.where(rng.random(rows) < 0.55, 0, rng.poisson(1.8, size=rows))

    process = np.where((idx % 4) == 0, "close_a", np.where((idx % 4) == 1, "close_b", np.where((idx % 4) == 2, "recon", "posting")))
    team = np.where(shift > 0.5, np.where((idx % 3) == 0, "ops", "fin"), np.where((idx % 3) == 0, "fin", "shared"))
    erp = np.where((idx % 2) == 0, "quorum", "quorum")

    return pd.DataFrame(
        {
            "event_ts": [v.isoformat() for v in ts],
            "queue_wait_mins": queue_wait,
            "duration_mins": duration,
            "service_runtime_mins": service_runtime,
            "eligible_jobs": eligible_jobs,
            "backlog_size": backlog,
            "metric_x": metric_x,
            "metric_y": metric_y,
            "event_count": event_count.astype(int),
            "retry_count": retry_count.astype(int),
            "process_id": process,
            "team": team,
            "erp_name": erp,
        }
    )


def _plugin_result_payload(result: Any) -> dict[str, Any]:
    payload = asdict(result)
    metrics = payload.get("metrics") or {}
    metrics.pop("runtime_ms", None)
    payload["metrics"] = metrics
    return _rounded_payload(payload)


def _rounded_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _rounded_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded_payload(v) for v in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            return str(value)
        return round(value, 10)
    return value


def _run_plugin_with_logging(tmp_path: Path, plugin_id: str, df: pd.DataFrame):
    run_dir = tmp_path / plugin_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "logs" / f"{plugin_id}.log"

    ctx = make_context(
        run_dir,
        df,
        settings={"seed": 1337, "allow_row_sampling": False},
        run_seed=1337,
    )

    def _logger(message: str) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{message}\n")

    ctx.logger = _logger
    module = import_module(f"plugins.{plugin_id}.plugin")
    return module.Plugin().run(ctx), log_path


def test_next30_handlers_and_scaffolds_exist() -> None:
    assert len(NEXT30_IDS) == 30
    for plugin_id in NEXT30_IDS:
        assert plugin_id in NEXT30_HANDLERS
        plugin_root = Path("plugins") / plugin_id
        assert (plugin_root / "plugin.py").exists()
        assert (plugin_root / "plugin.yaml").exists()
        assert (plugin_root / "config.schema.json").exists()
        assert (plugin_root / "output.schema.json").exists()


@pytest.mark.parametrize("plugin_id", NEXT30_IDS)
def test_next30_plugins_smoke_no_skip(tmp_path: Path, plugin_id: str) -> None:
    df = _next30_dataset()
    result, log_path = _run_plugin_with_logging(tmp_path, plugin_id, df)
    assert result.status == "ok", f"{plugin_id} -> {result.status}: {result.summary}"
    assert log_path.exists()
    assert log_path.read_text(encoding="utf-8").strip() != ""


@pytest.mark.parametrize("plugin_id", NEXT30_IDS)
def test_next30_ok_is_deterministic(tmp_path: Path, plugin_id: str) -> None:
    df = _next30_dataset()
    result_1, _ = _run_plugin_with_logging(tmp_path / "run1", plugin_id, df)
    result_2, _ = _run_plugin_with_logging(tmp_path / "run2", plugin_id, df)
    assert result_1.status == "ok"
    assert result_2.status == "ok"
    assert _plugin_result_payload(result_1) == _plugin_result_payload(result_2)
