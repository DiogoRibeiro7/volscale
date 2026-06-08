import pandas as pd
import pytest
from tests.benchmark_loader import load_atr_benchmark
from volscale.volatility import (
    compute_true_range,
    compute_atr,
    compute_mad,
    classify_volatility,
    compute_bollinger_bands,
    compute_keltner_channels,
    compute_donchian_channels,
)


def test_true_range():
    benchmark = load_atr_benchmark()
    high = pd.Series(benchmark["high"])
    low = pd.Series(benchmark["low"])
    close = pd.Series(benchmark["close"])
    result = compute_true_range(high, low, close)
    expected = pd.Series(benchmark["true_range"])
    pd.testing.assert_series_equal(result, expected)


def test_atr_ema():
    benchmark = load_atr_benchmark()
    high = pd.Series(benchmark["high"])
    low = pd.Series(benchmark["low"])
    close = pd.Series(benchmark["close"])
    expected = pd.Series(benchmark["atr_ema_3"])
    result = compute_atr(high, low, close, window=3, method="ema")
    pd.testing.assert_series_equal(result, expected)


def test_atr_sma_matches_reference_fixture():
    benchmark = load_atr_benchmark()
    high = pd.Series(benchmark["high"])
    low = pd.Series(benchmark["low"])
    close = pd.Series(benchmark["close"])
    expected = pd.Series(benchmark["atr_sma_3"])
    result = compute_atr(high, low, close, window=3, method="sma")
    pd.testing.assert_series_equal(result, expected)


def test_atr_rejects_unknown_method():
    high = pd.Series([10, 11, 12])
    low = pd.Series([9, 10, 11])
    close = pd.Series([9.5, 10.5, 11.5])
    with pytest.raises(ValueError, match="method"):
        compute_atr(high, low, close, window=2, method="wild")


def test_mad_constant():
    s = pd.Series([1, 1, 1, 1])
    result = compute_mad(s, window=2)
    expected = pd.Series([None, 0.0, 0.0, 0.0])
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_classify_volatility():
    vol = pd.Series([0.1, 0.2, 0.3, 0.4])
    labels = classify_volatility(vol, low_quantile=0.25, high_quantile=0.75)
    assert set(labels.unique()) == {"low", "medium", "high"}


def test_classify_volatility_rejects_bad_quantiles():
    vol = pd.Series([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="smaller"):
        classify_volatility(vol, low_quantile=0.8, high_quantile=0.2)


def test_bollinger_bands():
    s = pd.Series([1, 2, 3, 4, 5])
    bands = compute_bollinger_bands(s, window=3, num_std=1)
    assert bands.columns.tolist() == ["middle", "upper", "lower"]
    assert bands["upper"].iloc[2] > bands["middle"].iloc[2]


def test_keltner_channels():
    high = pd.Series([10, 11, 12, 13])
    low = pd.Series([9, 9.5, 10, 11])
    close = pd.Series([9.5, 10, 11, 12])
    channels = compute_keltner_channels(high, low, close, window=2, atr_multiplier=1)
    assert channels.columns.tolist() == ["middle", "upper", "lower"]
    assert channels["upper"].iloc[3] > channels["middle"].iloc[3]


def test_donchian_channels():
    high = pd.Series([10, 12, 13, 14])
    low = pd.Series([8, 9, 10, 11])
    channels = compute_donchian_channels(high, low, window=2)
    assert channels.columns.tolist() == ["middle", "upper", "lower"]
    assert channels["upper"].iloc[3] == 14
    assert channels["lower"].iloc[3] == 10
