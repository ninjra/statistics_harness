#!/usr/bin/env python3
from __future__ import annotations

# Back-compat shim: older plans/docs referenced this scaffolder name.
# The current implementation lives in `scripts/scaffold_top20_plugins.py`.

from scripts.scaffold_top20_plugins import main


if __name__ == "__main__":
    raise SystemExit(main())

