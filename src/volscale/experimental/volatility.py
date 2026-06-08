from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .._validation import (
    validate_finite_positive,
    validate_positive_int,
    validate_strictly_positive_series,
)


def compute_realized_volatility(prices: pd.Series, *, freq: str = "D") -> pd.Series:
    """Compute realized volatility from intraday prices."""
    prices = validate_strictly_positive_series(prices, "prices")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices index must be a pandas DatetimeIndex")
    log_returns = prices.div(prices.shift(1)).apply(np.log).dropna()
    return log_returns.pow(2).resample(freq).sum().pow(0.5)


def compute_garch_forecast(
    returns: pd.Series,
    horizon: int = 1,
    *,
    omega: float = 1e-6,
    alpha: float = 0.05,
    beta: float = 0.9,
) -> pd.Series:
    """Forecast volatility using a fixed-parameter GARCH-style recursion."""
    returns = returns.dropna().astype(float)
    if returns.empty:
        raise ValueError("returns series is empty")
    validate_positive_int(horizon, "horizon")
    if omega < 0:
        raise ValueError("omega must be non-negative")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative")
    if alpha + beta >= 1:
        raise ValueError("alpha + beta must be less than 1 for a stable forecast")

    variance = float(returns.var())  # type: ignore[arg-type]
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError("returns must have positive variance")

    prev_ret = float(returns.iloc[0])  # type: ignore[arg-type]
    for value in returns.iloc[1:]:
        variance = omega + alpha * prev_ret**2 + beta * variance
        prev_ret = float(value)  # type: ignore[arg-type]

    forecasts = []
    for _ in range(horizon):
        variance = omega + (alpha + beta) * variance
        forecasts.append(np.sqrt(variance))

    index = pd.RangeIndex(start=1, stop=horizon + 1, name="h")
    return pd.Series(forecasts, index=index, name="volatility")


def compute_implied_volatility(
    price: float,
    spot: float,
    strike: float,
    time: float,
    rate: float,
    option_type: str = "call",
    *,
    initial_vol: float = 0.2,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Estimate Black-Scholes implied volatility with bounded Newton steps."""
    validate_finite_positive(price, "price", allow_zero=True)
    validate_finite_positive(spot, "spot")
    validate_finite_positive(strike, "strike")
    validate_finite_positive(time, "time")
    if rate < -1:
        raise ValueError("rate is unrealistically low")
    validate_finite_positive(initial_vol, "initial_vol")
    validate_finite_positive(tol, "tol")
    validate_positive_int(max_iter, "max_iter")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    discount = math.exp(-rate * time)
    if option_type == "call":
        lower_bound = max(0.0, spot - strike * discount)
        upper_bound = spot
    else:
        lower_bound = max(0.0, strike * discount - spot)
        upper_bound = strike * discount
    if not (lower_bound <= price <= upper_bound):
        raise ValueError("price violates no-arbitrage bounds for the option inputs")

    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def norm_pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    def black_scholes(volatility: float) -> tuple[float, float]:
        d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * time) / (
            volatility * math.sqrt(time)
        )
        d2 = d1 - volatility * math.sqrt(time)

        if option_type == "call":
            model_price = spot * norm_cdf(d1) - strike * discount * norm_cdf(d2)
        else:
            model_price = strike * discount * norm_cdf(-d2) - spot * norm_cdf(-d1)

        vega = spot * math.sqrt(time) * norm_pdf(d1)
        return model_price, vega

    low_vol = 1e-9
    high_vol = max(initial_vol, 1.0)
    high_price, _ = black_scholes(high_vol)
    while high_price < price and high_vol < 10.0:
        high_vol *= 2.0
        high_price, _ = black_scholes(high_vol)
    if high_price < price:
        raise RuntimeError("failed to bracket implied volatility")

    volatility = min(max(initial_vol, low_vol), high_vol)
    for _ in range(max_iter):
        model_price, vega = black_scholes(volatility)
        diff = model_price - price
        if abs(diff) < tol:
            return volatility

        if vega > 1e-8:
            candidate = volatility - diff / vega
            if low_vol < candidate < high_vol:
                volatility = candidate
            else:
                volatility = 0.5 * (low_vol + high_vol)
        else:
            volatility = 0.5 * (low_vol + high_vol)

        model_price, _ = black_scholes(volatility)
        if model_price > price:
            high_vol = volatility
        else:
            low_vol = volatility

    raise RuntimeError("implied volatility did not converge")


def compute_regime_probabilities(returns: pd.Series, n_iter: int = 10) -> pd.DataFrame:
    """Estimate two-state volatility regimes with a simple Gaussian HMM."""
    validate_positive_int(n_iter, "n_iter")
    returns = returns.dropna().astype(float)
    if returns.empty:
        raise ValueError("returns series is empty")

    values = returns.to_numpy()
    count = len(values)

    def logsumexp(array: np.ndarray) -> float:
        array_max = np.max(array)
        return array_max + np.log(np.exp(array - array_max).sum())

    base_sigma = float(np.std(values))
    if not math.isfinite(base_sigma) or base_sigma <= 0:
        raise ValueError("returns must have positive variance")

    eps = 1e-8
    sigma_low = max(base_sigma * 0.5, eps)
    sigma_high = max(base_sigma * 2.0, eps)
    transition = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=float)
    priors = np.array([0.5, 0.5], dtype=float)
    prev_log_likelihood = -np.inf

    for _ in range(n_iter):
        log_likelihoods = np.vstack(
            [
                -0.5 * ((values / sigma_low) ** 2 + np.log(2 * np.pi * sigma_low**2)),
                -0.5 * ((values / sigma_high) ** 2 + np.log(2 * np.pi * sigma_high**2)),
            ]
        ).T

        log_alpha = np.zeros((count, 2))
        log_alpha[0] = np.log(priors) + log_likelihoods[0]
        for t in range(1, count):
            for state in range(2):
                log_alpha[t, state] = log_likelihoods[t, state] + logsumexp(
                    log_alpha[t - 1] + np.log(transition[:, state])
                )

        log_beta = np.zeros((count, 2))
        for t in range(count - 2, -1, -1):
            for state in range(2):
                log_beta[t, state] = logsumexp(
                    np.log(transition[state]) + log_likelihoods[t + 1] + log_beta[t + 1]
                )

        log_gamma = log_alpha + log_beta
        normalization = np.apply_along_axis(logsumexp, 1, log_gamma)
        gamma = np.exp(log_gamma - normalization[:, None])

        log_likelihood = float(normalization.sum())
        if log_likelihood < prev_log_likelihood - 1e-6:
            raise RuntimeError("regime estimation became numerically unstable")
        prev_log_likelihood = log_likelihood

        log_xi = np.zeros((count - 1, 2, 2))
        for t in range(count - 1):
            for i in range(2):
                for j in range(2):
                    log_xi[t, i, j] = (
                        log_alpha[t, i]
                        + np.log(transition[i, j])
                        + log_likelihoods[t + 1, j]
                        + log_beta[t + 1, j]
                    )
            log_xi[t] -= logsumexp(log_xi[t].ravel())
        xi = np.exp(log_xi)

        priors = gamma[0]
        transition_den = np.maximum(gamma[:-1].sum(axis=0)[:, None], eps)
        transition = np.clip(xi.sum(axis=0) / transition_den, eps, 1.0)
        transition = transition / transition.sum(axis=1, keepdims=True)
        sigma_low = max(
            math.sqrt((gamma[:, 0] * values**2).sum() / max(gamma[:, 0].sum(), eps)),
            eps,
        )
        sigma_high = max(
            math.sqrt((gamma[:, 1] * values**2).sum() / max(gamma[:, 1].sum(), eps)),
            eps,
        )

    return pd.DataFrame(gamma, index=returns.index, columns=["low", "high"])
