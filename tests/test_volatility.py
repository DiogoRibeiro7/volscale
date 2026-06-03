import pandas as pd
import pytest
from volnorm.volatility import (
    compute_true_range,
    compute_atr,
    compute_mad,
    classify_volatility,
    compute_bollinger_bands,
    compute_keltner_channels,
    compute_donchian_channels,
    compute_realized_volatility,
    compute_garch_forecast,
    compute_implied_volatility,
    compute_regime_probabilities,
)
import numpy as np
import math


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


def test_realized_volatility():
    idx = pd.DatetimeIndex(
        [
            "2024-01-01 09:30",
            "2024-01-01 09:31",
            "2024-01-01 09:32",
            "2024-01-02 09:30",
            "2024-01-02 09:31",
            "2024-01-02 09:32",
        ]
    )
    prices = pd.Series([100, 101, 102, 103, 104, 105], index=idx)

    result = compute_realized_volatility(prices)

    log_returns = (prices / prices.shift(1)).apply(np.log).dropna()
    expected = log_returns.pow(2).resample("D").sum().pow(0.5)

    pd.testing.assert_series_equal(result, expected)


def test_realized_volatility_matches_manual_daily_benchmark():
    idx = pd.DatetimeIndex(
        [
            "2024-01-01 09:30",
            "2024-01-01 09:31",
            "2024-01-01 09:32",
            "2024-01-02 09:30",
            "2024-01-02 09:31",
        ]
    )
    prices = pd.Series(
        [
            100.0,
            100.0 * math.exp(0.1),
            100.0 * math.exp(0.1 + 0.2),
            100.0 * math.exp(0.1 + 0.2 + 0.3),
            100.0 * math.exp(0.1 + 0.2 + 0.3 + 0.4),
        ],
        index=idx,
    )
    result = compute_realized_volatility(prices)
    expected = pd.Series(
        [math.sqrt(0.1**2 + 0.2**2), math.sqrt(0.3**2 + 0.4**2)],
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    pd.testing.assert_series_equal(result, expected)


def test_realized_volatility_requires_datetime_index():
    prices = pd.Series([100, 101, 102])
    with pytest.raises(TypeError, match="DatetimeIndex"):
        compute_realized_volatility(prices)


def test_garch_forecast():
    returns = pd.Series([0.01, -0.02, 0.015, -0.005])
    forecast = compute_garch_forecast(returns, horizon=2)
    assert len(forecast) == 2
    assert forecast.index.tolist() == [1, 2]
    assert (forecast > 0).all()


def test_garch_forecast_long_horizon_reverts_toward_unconditional_volatility():
    returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.012, -0.008])
    omega = 1e-6
    alpha = 0.05
    beta = 0.9
    forecast = compute_garch_forecast(
        returns,
        horizon=200,
        omega=omega,
        alpha=alpha,
        beta=beta,
    )
    long_run_vol = math.sqrt(omega / (1 - alpha - beta))
    assert abs(float(forecast.iloc[-1]) - long_run_vol) < 5e-4


def test_garch_forecast_rejects_unstable_parameters():
    returns = pd.Series([0.01, -0.02, 0.015, -0.005])
    with pytest.raises(ValueError, match="less than 1"):
        compute_garch_forecast(returns, alpha=0.4, beta=0.7)


def test_implied_volatility():
    spot = 100.0
    strike = 105.0
    time = 0.5
    rate = 0.01
    true_vol = 0.2

    # Black-Scholes formula for a call option
    d1 = (np.log(spot / strike) + (rate + 0.5 * true_vol**2) * time) / (
        true_vol * np.sqrt(time)
    )
    d2 = d1 - true_vol * np.sqrt(time)

    def cdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / np.sqrt(2)))

    price = spot * cdf(d1) - strike * np.exp(-rate * time) * cdf(d2)

    est = compute_implied_volatility(
        price, spot, strike, time, rate, option_type="call"
    )
    assert abs(est - true_vol) < 1e-4


def test_implied_volatility_matches_known_black_scholes_benchmark():
    est = compute_implied_volatility(
        price=10.4506,
        spot=100.0,
        strike=100.0,
        time=1.0,
        rate=0.05,
        option_type="call",
    )
    assert abs(est - 0.2) < 1e-4


def test_implied_volatility_rejects_arbitrage_violations():
    with pytest.raises(ValueError, match="no-arbitrage"):
        compute_implied_volatility(
            price=200.0,
            spot=100.0,
            strike=105.0,
            time=0.5,
            rate=0.01,
        )


def test_regime_probabilities_identify_high_low():
    np.random.seed(0)
    low = np.random.normal(0, 0.01, size=30)
    high = np.random.normal(0, 0.05, size=30)
    returns = pd.Series(np.concatenate([low, high, low]))

    probs = compute_regime_probabilities(returns, n_iter=5)

    assert probs.loc[:29, "low"].mean() > 0.5
    assert probs.loc[30:59, "high"].mean() > 0.5


def test_regime_probabilities_reject_constant_series():
    returns = pd.Series([0.0] * 10)
    with pytest.raises(ValueError, match="positive variance"):
        compute_regime_probabilities(returns)
