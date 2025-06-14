import pandas as pd
import numpy as np
from volnorm.rolling import (
    compute_rolling_std,
    compute_sma,
    compute_atr_proxy,
    compute_ema,
    compute_rolling_max,
    compute_rolling_min,
    compute_zscore,
    compute_wma,
)


def test_rolling_std():
    s = pd.Series([1, 2, 3, 4, 5])
    result = compute_rolling_std(s, window=3)
    assert result.isna().sum() == 2  # first two should be NaN


def test_rolling_std_constant():
    s = pd.Series([5, 5, 5, 5])
    result = compute_rolling_std(s, window=2)
    expected = pd.Series([None, 0.0, 0.0, 0.0])
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_sma():
    s = pd.Series([1, 2, 3, 4, 5])
    result = compute_sma(s, window=3)
    assert result.iloc[2] == 2.0
    assert result.iloc[4] == 4.0


def test_atr_proxy():
    s = pd.Series([1, 4, 3, 2, 5])
    result = compute_atr_proxy(s, window=3)
    expected = pd.Series([None, None, 3.0, 2.0, 3.0])
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_ema():
    s = pd.Series([1, 2, 3, 4, 5])
    result = compute_ema(s, span=3)
    expected = s.ewm(span=3, adjust=False).mean()
    pd.testing.assert_series_equal(result, expected)


def test_ema_partial_value():
    s = pd.Series([1, 2, 3, 4, 5])
    result = compute_ema(s, span=3)
    assert np.isclose(result.iloc[2], 2.25)


def test_rolling_max_min():
    s = pd.Series([1, 2, 3, 2, 1])
    max_result = compute_rolling_max(s, window=3)
    min_result = compute_rolling_min(s, window=3)
    assert max_result.iloc[4] == 3
    assert min_result.iloc[4] == 1


def test_rolling_max_min_constant():
    s = pd.Series([5, 5, 5, 5])
    max_result = compute_rolling_max(s, window=2)
    min_result = compute_rolling_min(s, window=2)
    expected = pd.Series([None, 5.0, 5.0, 5.0])
    pd.testing.assert_series_equal(max_result, expected, check_names=False)
    pd.testing.assert_series_equal(min_result, expected, check_names=False)


def test_zscore():
    s = pd.Series([1, 2, 3, 4, 5])
    result = compute_zscore(s, window=3)
    expected = (s - s.rolling(window=3, min_periods=3).mean()) / s.rolling(
        window=3, min_periods=3
    ).std()
    pd.testing.assert_series_equal(result, expected)


def test_zscore_constant():
    s = pd.Series([5, 5, 5, 5])
    result = compute_zscore(s, window=2)
    assert result.isna().all()


def test_wma():
    s = pd.Series([1, 2, 3, 4, 5])
    result = compute_wma(s, window=3)
    weights = np.arange(1, 4)
    expected = s.rolling(window=3, min_periods=3).apply(
        lambda x: np.dot(x, weights) / weights.sum(),
        raw=True,
    )
    pd.testing.assert_series_equal(result, expected)
