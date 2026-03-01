from __future__ import annotations

from statistic_harness.core.run_context import RunContext


def test_run_context_child_seed_is_deterministic() -> None:
    ctx = RunContext(run_id="r1", run_seed=42)
    assert ctx.child_seed("plugin_a") == ctx.child_seed("plugin_a")
    assert ctx.child_seed("plugin_a") != ctx.child_seed("plugin_b")


def test_run_context_rng_is_deterministic() -> None:
    ctx = RunContext(run_id="r2", run_seed=99)
    a = [ctx.rng("x").random() for _ in range(3)]
    b = [ctx.rng("x").random() for _ in range(3)]
    assert a == b

