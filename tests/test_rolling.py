import pandas as pd
from volnorm.rolling import compute_rolling_std, compute_sma, compute_atr_proxy

def test_rolling_std():
    s = pd.Series([1, 2, 3, 4, 5])
    result = compute_rolling_std(s, window=3)
    assert result.isna().sum() == 2  # first two should be NaN

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
