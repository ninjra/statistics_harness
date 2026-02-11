"""Legacy/shim CLI module.

Docs historically referenced `src/statistic_harness/core/cli.py`, but the
actual entrypoint lives in `src/statistic_harness/cli.py`.

This shim exists to keep documentation links stable and to provide a safe
import target for tooling that expects the old path.
"""

from __future__ import annotations

from statistic_harness.cli import main

__all__ = ["main"]

