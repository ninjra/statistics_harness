from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from statistic_harness.core.stat_plugins.next30b_addon import (
    HANDLERS as NEXT30B_HANDLERS,
    NEXT30B_IDS,
)
from tests.conftest import make_context


def _next30b_dataset(rows: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(20260213)
    base = datetime(2025, 1, 1, 0, 0, 0)
    idx = np.arange(rows)
    half = rows // 2
    shift = (idx >= half).astype(float)

    jitter = rng.integers(0, 40, size=rows, dtype=np.int64)
    event_ts = [base + timedelta(minutes=int(i * 45 + jitter[i])) for i in idx]

    seasonal = np.sin((2.0 * np.pi * idx) / 24.0)
    noise = rng.normal(0.0, 0.5, size=rows)

    queue_wait = 15.0 + 3.0 * seasonal + 6.5 * shift + noise
    duration = 35.0 + 2.4 * queue_wait + 4.0 * shift + rng.normal(0.0, 1.8, size=rows)
    service_runtime = 20.0 + 1.4 * queue_wait + rng.normal(0.0, 1.2, size=rows)
    throughput = 120.0 + 0.8 * queue_wait - 3.0 * shift + rng.normal(0.0, 3.0, size=rows)
    cpu_load = 0.5 + 0.03 * queue_wait + rng.normal(0.0, 0.04, size=rows)
    metric_x = 10.0 + 0.6 * seasonal + rng.normal(0.0, 0.6, size=rows)
    metric_y = 8.0 + 0.7 * metric_x + rng.normal(0.0, 0.35, size=rows)
    metric_z = 4.0 + 0.3 * metric_x - 0.2 * metric_y + rng.normal(0.0, 0.25, size=rows)

    raw_count = rng.poisson(np.where(shift > 0.5, 7.0, 3.0), size=rows)
    structural_zero = rng.random(rows) < 0.30
    event_count = np.where(structural_zero, 0, raw_count)
    queue_count = np.where(rng.random(rows) < 0.45, 0, rng.poisson(2.0, size=rows))
    binary_outcome = (rng.random(rows) < np.where(shift > 0.5, 0.62, 0.38)).astype(int)

    # Directed edges with structural shift in post half.
    node_count = 24
    src = [f"n{int(i % node_count):02d}" for i in idx]
    dst = []
    for i in idx:
        if i < half:
            dst_idx = (int(i) + 3) % node_count
        else:
            dst_idx = (int(i) + 7 + (int(i) % 3)) % node_count
        dst.append(f"n{dst_idx:02d}")

    event_type = np.where((idx % 5) == 0, "timeout", np.where((idx % 5) == 1, "rework", np.where((idx % 5) == 2, "success", np.where((idx % 5) == 3, "failure", "cancelled"))))
    process = np.where((idx % 4) == 0, "close_a", np.where((idx % 4) == 1, "close_b", np.where((idx % 4) == 2, "recon", "posting")))
    team = np.where(shift > 0.5, np.where((idx % 3) == 0, "ops", "fin"), np.where((idx % 3) == 0, "fin", "shared"))
    erp = np.where((idx % 2) == 0, "quorum", "quorum")

    df = pd.DataFrame(
        {
            "event_ts": [v.isoformat() for v in event_ts],
            "queue_wait_mins": queue_wait,
            "duration_mins": duration,
            "service_runtime_mins": service_runtime,
            "throughput_count": throughput,
            "cpu_load": cpu_load,
            "metric_x": metric_x,
            "metric_y": metric_y,
            "metric_z": metric_z,
            "event_count": event_count.astype(int),
            "queue_count": queue_count.astype(int),
            "binary_outcome": binary_outcome.astype(int),
            "status_type": event_type,
            "process_id": process,
            "team": team,
            "erp_name": erp,
            "src_node": src,
            "dst_node": dst,
        }
    )
    return df


def _normalize_result_payload(result: Any) -> dict[str, Any]:
    payload = asdict(result)
    metrics = payload.get("metrics") or {}
    metrics.pop("runtime_ms", None)
    payload["metrics"] = metrics
    return _round_payload(payload)


def _round_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _round_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_payload(v) for v in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            return str(value)
        return round(value, 9)
    return value


def _run_plugin_with_logging(tmp_path: Path, plugin_id: str, df: pd.DataFrame):
    run_dir = tmp_path / plugin_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "logs" / f"{plugin_id}.log"

    settings = {
        "seed": 1337,
        "allow_row_sampling": False,
        "plugin": {
            "max_points_for_quadratic": 2000,
            "max_resamples": 100,
        },
    }
    ctx = make_context(run_dir, df, settings=settings, run_seed=1337)

    def _logger(message: str) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{message}\n")

    ctx.logger = _logger
    module = import_module(f"plugins.{plugin_id}.plugin")
    return module.Plugin().run(ctx), log_path


def test_next30b_handlers_and_scaffolds_exist() -> None:
    assert len(NEXT30B_IDS) == 30
    for plugin_id in NEXT30B_IDS:
        assert plugin_id in NEXT30B_HANDLERS
        root = Path("plugins") / plugin_id
        assert (root / "plugin.py").exists()
        assert (root / "plugin.yaml").exists()
        assert (root / "config.schema.json").exists()
        assert (root / "output.schema.json").exists()


@pytest.mark.parametrize("plugin_id", NEXT30B_IDS)
def test_next30b_plugins_smoke_no_skip(tmp_path: Path, plugin_id: str) -> None:
    df = _next30b_dataset()
    result, log_path = _run_plugin_with_logging(tmp_path, plugin_id, df)
    assert result.status == "ok", f"{plugin_id} -> {result.status}: {result.summary}"
    assert not str(result.summary).startswith("No actionable result"), f"{plugin_id} returned non-actionable result: {result.summary}"
    assert log_path.exists()
    assert log_path.read_text(encoding="utf-8").strip() != ""


@pytest.mark.parametrize("plugin_id", NEXT30B_IDS)
def test_next30b_ok_is_deterministic(tmp_path: Path, plugin_id: str) -> None:
    df = _next30b_dataset()
    result_1, _ = _run_plugin_with_logging(tmp_path / "run1", plugin_id, df)
    result_2, _ = _run_plugin_with_logging(tmp_path / "run2", plugin_id, df)
    assert result_1.status == "ok", f"{plugin_id} -> {result_1.status}: {result_1.summary}"
    assert result_2.status == "ok", f"{plugin_id} -> {result_2.status}: {result_2.summary}"
    assert not str(result_1.summary).startswith("No actionable result"), f"{plugin_id} returned non-actionable result on run1"
    assert not str(result_2.summary).startswith("No actionable result"), f"{plugin_id} returned non-actionable result on run2"
    assert _normalize_result_payload(result_1) == _normalize_result_payload(result_2)
