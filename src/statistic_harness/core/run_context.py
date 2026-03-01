from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any

from .utils import stable_hash


@dataclass(frozen=True)
class RunContext:
    """Deterministic run-scoped seed helper."""

    run_id: str
    run_seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def child_seed(self, name: str) -> int:
        token = f"{self.run_seed}:{self.run_id}:{name}"
        return int(stable_hash(token))

    def rng(self, name: str) -> random.Random:
        return random.Random(self.child_seed(name))

