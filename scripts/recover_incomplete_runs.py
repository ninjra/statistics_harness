#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

from statistic_harness.core.pipeline import Pipeline


def main() -> int:
    # This script exists to quickly mark stale "running" runs as aborted when their recorded PID
    # is not alive. Running a full SQLite integrity check on multi-GB DBs defeats the purpose.
    os.environ.setdefault("STAT_HARNESS_STARTUP_INTEGRITY", "off")
    # Constructing Pipeline triggers startup recovery logic.
    Pipeline(Path("appdata"), Path("plugins"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
