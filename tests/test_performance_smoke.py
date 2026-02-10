import os
import time
from pathlib import Path

from statistic_harness.core.pipeline import Pipeline


def test_performance_smoke(tmp_path, monkeypatch):
    appdata = tmp_path / "appdata"
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata))
    pipeline = Pipeline(appdata, Path("plugins"))

    start = time.perf_counter()
    pipeline.run(Path("tests/fixtures/synth_linear.csv"), ["profile_basic"], {}, 123)
    elapsed = time.perf_counter() - start

    # The harness runs multiple plugins in separate subprocesses (ingest + normalize + profile + report),
    # each of which imports heavy scientific packages. On slower environments (WSL/AV scanning),
    # 15s is too tight and causes flakes. Override in CI if you can reliably hit a lower bound.
    max_seconds = float(os.environ.get("STAT_HARNESS_PERF_MAX_SECONDS", "25"))
    assert elapsed < max_seconds
