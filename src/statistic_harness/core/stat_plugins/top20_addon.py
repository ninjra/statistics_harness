from __future__ import annotations

"""Top20 add-on pack (reserved module).

Historical plans referenced a stat-plugin style implementation for the Top20
methods pack. The production implementation is intentionally SQL-first and lives
in `src/statistic_harness/core/top20_plugins.py`, with direct plugin wrappers
under `plugins/analysis_*_v1/`.

This module exists to keep documentation/binding references valid and to provide
an obvious home if we later decide to expose Top20 algorithms as stat-plugin
handlers.
"""

from typing import Any, Callable

from statistic_harness.core.types import PluginResult


HANDLERS: dict[str, Callable[..., PluginResult]] = {}

