import math

import numpy as np
import pandas as pd
import pytest
from tests.benchmark_loader import (
    load_black_scholes_benchmarks,
    load_realized_volatility_benchmark,
)

from volscale.experimental.volatility import (
    compute_garch_forecast,
    compute_implied_volatility,
    compute_realized_volatility,
    compute_regime_probabilities,
)


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
    log_returns = prices.div(prices.shift(1)).apply(np.log).dropna()
    expected = log_returns.pow(2).resample("D").sum().pow(0.5)

    pd.testing.assert_series_equal(result, expected)


def test_realized_volatility_matches_manual_daily_benchmark():
    benchmark = load_realized_volatility_benchmark()
    prices = pd.Series(
        [
            100.0,
            100.0 * math.exp(0.1),
            100.0 * math.exp(0.1 + 0.2),
            100.0 * math.exp(0.1 + 0.2 + 0.3),
            100.0 * math.exp(0.1 + 0.2 + 0.3 + 0.4),
        ],
        index=benchmark["index"],
    )
    result = compute_realized_volatility(prices)
    pd.testing.assert_series_equal(result, benchmark["expected_daily"])


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


@pytest.mark.parametrize("case", load_black_scholes_benchmarks())
def test_implied_volatility_matches_reference_benchmarks(case):
    est = compute_implied_volatility(
        price=case["price"],
        spot=case["spot"],
        strike=case["strike"],
        time=case["time"],
        rate=case["rate"],
        option_type=case["option_type"],
    )
    assert abs(est - case["volatility"]) < 1e-4


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
