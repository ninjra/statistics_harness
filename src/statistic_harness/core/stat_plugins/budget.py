from __future__ import annotations

import time


class BudgetTimer:
    def __init__(self, time_budget_ms: int | None) -> None:
        self.time_budget_ms = time_budget_ms
        self.started_at = time.monotonic()

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.started_at) * 1000.0

    def remaining_ms(self) -> float | None:
        if self.time_budget_ms is None:
            return None
        return max(0.0, float(self.time_budget_ms) - self.elapsed_ms())

    def exceeded(self) -> bool:
        if self.time_budget_ms is None:
            return False
        return self.elapsed_ms() > float(self.time_budget_ms)

    def ensure(self, message: str = "time_budget_exceeded") -> None:
        if self.exceeded():
            raise TimeoutError(message)
