import numpy as np
import pandas as pd
from statistic_harness.core.lever_library import dynamic_threshold


def test_dynamic_threshold_basic():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = dynamic_threshold(values)
    expected = float(np.median(values) + 1.5 * np.std(values))
    assert abs(result - expected) < 1e-9


def test_dynamic_threshold_custom_multiplier():
    values = np.array([10.0, 20.0, 30.0])
    result = dynamic_threshold(values, multiplier=2.0)
    expected = float(np.median(values) + 2.0 * np.std(values))
    assert abs(result - expected) < 1e-9


def test_dynamic_threshold_empty_returns_fallback():
    result = dynamic_threshold(np.array([]), fallback=42.0)
    assert result == 42.0


def test_dynamic_threshold_empty_no_fallback_returns_inf():
    result = dynamic_threshold(np.array([]))
    assert result == float("inf")


def test_dynamic_threshold_handles_nans():
    values = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    result = dynamic_threshold(values)
    clean = np.array([1.0, 3.0, 5.0])
    expected = float(np.median(clean) + 1.5 * np.std(clean))
    assert abs(result - expected) < 1e-9


def test_dynamic_threshold_with_pandas_series():
    s = pd.Series([2.0, 4.0, 6.0, 8.0])
    result = dynamic_threshold(s)
    arr = s.values
    expected = float(np.median(arr) + 1.5 * np.std(arr))
    assert abs(result - expected) < 1e-9
