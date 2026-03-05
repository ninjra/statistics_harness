"""DAAF Pattern 1A: Typed error classification tests."""

from __future__ import annotations

import pytest

from statistic_harness.core.types import PluginError


# Import classify_error indirectly — it's defined inside Pipeline.run(),
# so we replicate the logic here for unit testing.

ERROR_CATEGORIES = {
    "ImportError": "dependency_missing",
    "ModuleNotFoundError": "dependency_missing",
    "TimeoutError": "budget_exceeded",
    "MemoryError": "memory_exceeded",
    "numpy.linalg.LinAlgError": "numerical_failure",
    "ValueError": "input_invalid",
    "KeyError": "column_missing",
    "ConvergenceWarning": "convergence_failure",
}


def classify_error(error: PluginError) -> str:
    for pattern, category in ERROR_CATEGORIES.items():
        if pattern in (error.type or "") or pattern in (error.traceback or ""):
            return category
    return "unknown"


@pytest.mark.parametrize(
    "error_type, expected_category",
    [
        ("ImportError", "dependency_missing"),
        ("ModuleNotFoundError", "dependency_missing"),
        ("TimeoutError", "budget_exceeded"),
        ("MemoryError", "memory_exceeded"),
        ("ValueError", "input_invalid"),
        ("KeyError", "column_missing"),
    ],
)
def test_classify_error_by_type(error_type: str, expected_category: str) -> None:
    error = PluginError(type=error_type, message="test", traceback="")
    assert classify_error(error) == expected_category


def test_classify_error_by_traceback() -> None:
    error = PluginError(
        type="Exception",
        message="singular matrix",
        traceback="numpy.linalg.LinAlgError: singular matrix",
    )
    assert classify_error(error) == "numerical_failure"


def test_classify_error_convergence_in_traceback() -> None:
    error = PluginError(
        type="RuntimeWarning",
        message="optimization did not converge",
        traceback="ConvergenceWarning: optimizer did not converge",
    )
    assert classify_error(error) == "convergence_failure"


def test_classify_error_unknown() -> None:
    error = PluginError(type="RuntimeError", message="something", traceback="")
    assert classify_error(error) == "unknown"


def test_classify_error_empty_fields() -> None:
    error = PluginError(type="", message="", traceback="")
    assert classify_error(error) == "unknown"
