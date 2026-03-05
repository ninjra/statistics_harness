"""DAAF Pattern 1C: Run-level signals aggregation tests."""

from __future__ import annotations

import json


def _aggregate_signals(plugin_results: list[dict]) -> list[dict]:
    """Replicate the signals aggregation logic from pipeline.py for testing."""
    signals: list[dict] = []
    for row in plugin_results:
        pid = str(row.get("plugin_id") or "")
        status = str(row.get("status") or "").lower()
        if status != "ok":
            continue
        mdata = row.get("metrics", {})
        if isinstance(mdata, str):
            mdata = json.loads(mdata)
        bdata = row.get("budget", {})
        if isinstance(bdata, str):
            bdata = json.loads(bdata)
        if bdata.get("sampled"):
            signals.append({
                "type": "sampled_input",
                "plugin_id": pid,
                "note": "Result based on sampled data, not full dataset",
            })
        n = mdata.get("n_observations")
        if isinstance(n, (int, float)) and 30 <= n <= 50:
            signals.append({
                "type": "marginal_sample_size",
                "plugin_id": pid,
                "note": f"n={n}, just above minimum",
            })
        debug = row.get("debug", {})
        if isinstance(debug, str) and "convergence" in debug.lower():
            signals.append({"type": "convergence_concern", "plugin_id": pid})
        elif isinstance(debug, dict) and "convergence" in str(debug).lower():
            signals.append({"type": "convergence_concern", "plugin_id": pid})
    return signals


def test_sampled_input_signal() -> None:
    results = [
        {
            "plugin_id": "plugin_a",
            "status": "ok",
            "metrics": {"n_observations": 1000},
            "budget": {"sampled": True},
            "debug": {},
        }
    ]
    signals = _aggregate_signals(results)
    assert any(s["type"] == "sampled_input" and s["plugin_id"] == "plugin_a" for s in signals)


def test_marginal_sample_size_signal() -> None:
    results = [
        {
            "plugin_id": "plugin_b",
            "status": "ok",
            "metrics": {"n_observations": 35},
            "budget": {},
            "debug": {},
        }
    ]
    signals = _aggregate_signals(results)
    assert any(s["type"] == "marginal_sample_size" and s["plugin_id"] == "plugin_b" for s in signals)


def test_no_signal_for_healthy_plugin() -> None:
    results = [
        {
            "plugin_id": "plugin_c",
            "status": "ok",
            "metrics": {"n_observations": 500},
            "budget": {"sampled": False},
            "debug": {},
        }
    ]
    signals = _aggregate_signals(results)
    assert signals == []


def test_convergence_concern_signal() -> None:
    results = [
        {
            "plugin_id": "plugin_d",
            "status": "ok",
            "metrics": {},
            "budget": {},
            "debug": {"warnings": ["ConvergenceWarning: did not converge"]},
        }
    ]
    signals = _aggregate_signals(results)
    assert any(s["type"] == "convergence_concern" and s["plugin_id"] == "plugin_d" for s in signals)


def test_skipped_plugins_excluded() -> None:
    results = [
        {
            "plugin_id": "plugin_e",
            "status": "skipped",
            "metrics": {"n_observations": 35},
            "budget": {"sampled": True},
            "debug": {},
        }
    ]
    signals = _aggregate_signals(results)
    assert signals == []


def test_multiple_signals_from_same_plugin() -> None:
    results = [
        {
            "plugin_id": "plugin_f",
            "status": "ok",
            "metrics": {"n_observations": 40},
            "budget": {"sampled": True},
            "debug": {"note": "convergence issue"},
        }
    ]
    signals = _aggregate_signals(results)
    types = {s["type"] for s in signals}
    assert "sampled_input" in types
    assert "marginal_sample_size" in types
    assert "convergence_concern" in types
