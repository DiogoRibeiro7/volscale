import pandas as pd
from volnorm.volatility import (
    compute_true_range,
    compute_atr,
    compute_mad,
    classify_volatility,
)


def test_true_range():
    high = pd.Series([10, 11, 12])
    low = pd.Series([8, 9, 10])
    close = pd.Series([9, 10, 11])
    result = compute_true_range(high, low, close)
    expected = pd.Series([2.0, 2.0, 2.0])
    pd.testing.assert_series_equal(result, expected)


def test_atr_ema():
    high = pd.Series([10, 11, 12, 13])
    low = pd.Series([9, 10, 11, 12])
    close = pd.Series([9.5, 10.5, 11.5, 12.5])
    expected = compute_true_range(high, low, close).ewm(span=2, adjust=False).mean()
    result = compute_atr(high, low, close, window=2)
    pd.testing.assert_series_equal(result, expected)


def test_mad_constant():
    s = pd.Series([1, 1, 1, 1])
    result = compute_mad(s, window=2)
    expected = pd.Series([None, 0.0, 0.0, 0.0])
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_classify_volatility():
    vol = pd.Series([0.1, 0.2, 0.3, 0.4])
    labels = classify_volatility(vol, low_quantile=0.25, high_quantile=0.75)
    assert set(labels.unique()) == {"low", "medium", "high"}
