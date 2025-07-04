import math
import numpy as np
import pandas as pd
from .rolling import compute_sma, compute_rolling_std


def compute_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Return the True Range for each period."""
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    return tr_components.max(axis=1)


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
    *,
    method: str = "ema",
) -> pd.Series:
    """Average True Range calculated via EMA or SMA."""
    tr = compute_true_range(high, low, close)
    if method == "ema":
        return tr.ewm(span=window, adjust=False).mean()
    return tr.rolling(window=window, min_periods=window).mean()


def compute_mad(values: pd.Series, window: int) -> pd.Series:
    """Rolling Median Absolute Deviation."""
    return values.rolling(window=window, min_periods=window).apply(
        lambda x: np.median(np.abs(x - np.median(x)))
    )


def classify_volatility(
    volatility: pd.Series,
    low_quantile: float = 0.25,
    high_quantile: float = 0.75,
) -> pd.Series:
    """Classify volatility levels using quantile thresholds."""
    low_thresh = volatility.quantile(low_quantile)
    high_thresh = volatility.quantile(high_quantile)

    def _label(v: float) -> str:
        if pd.isna(v):
            return "unknown"
        if v < low_thresh:
            return "low"
        if v > high_thresh:
            return "high"
        return "medium"

    return volatility.apply(_label)


def compute_bollinger_bands(
    prices: pd.Series, window: int, *, num_std: float = 2.0
) -> pd.DataFrame:
    """Return Bollinger Bands as a DataFrame."""
    sma = compute_sma(prices, window)
    std = compute_rolling_std(prices, window)
    upper = sma + num_std * std
    lower = sma - num_std * std
    return pd.DataFrame({"middle": sma, "upper": upper, "lower": lower})


def compute_keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
    *,
    atr_multiplier: float = 2.0,
) -> pd.DataFrame:
    """Return Keltner Channels using EMA and ATR."""
    ema = close.ewm(span=window, adjust=False).mean()
    atr = compute_atr(high, low, close, window)
    upper = ema + atr_multiplier * atr
    lower = ema - atr_multiplier * atr
    return pd.DataFrame({"middle": ema, "upper": upper, "lower": lower})


def compute_donchian_channels(
    high: pd.Series, low: pd.Series, window: int
) -> pd.DataFrame:
    """Return Donchian Channels using rolling extremes."""
    upper = high.rolling(window=window, min_periods=window).max()
    lower = low.rolling(window=window, min_periods=window).min()
    middle = (upper + lower) / 2
    return pd.DataFrame({"middle": middle, "upper": upper, "lower": lower})


def compute_realized_volatility(prices: pd.Series, *, freq: str = "D") -> pd.Series:
    """Compute realized volatility from intraday prices.

    Args:
        prices: Series of prices indexed by a ``DatetimeIndex`` at intraday frequency.
        freq: Frequency string for resampling the output, defaults to daily (``"D"``).

    Returns:
        Realized volatility aggregated at the specified frequency.
    """

    # Compute log returns and drop the initial NaN introduced by the shift. Using
    # ``apply`` keeps the pandas ``Series`` type so static type checkers do not
    # infer an ``ndarray`` from ``numpy`` operations.
    log_returns = (prices / prices.shift(1)).apply(np.log).dropna()

    # Sum squared returns within each resampling window and take the square root
    return log_returns.pow(2).resample(freq).sum().pow(0.5)


def compute_garch_forecast(
    returns: pd.Series,
    horizon: int = 1,
    *,
    omega: float = 1e-6,
    alpha: float = 0.05,
    beta: float = 0.9,
) -> pd.Series:
    """Forecast volatility using a simple GARCH(1,1) model.

    This function implements an unparameterized GARCH(1,1) recursion. It
    estimates conditional variance based on the input returns and projects it
    forward ``horizon`` steps. The parameters ``omega``, ``alpha`` and ``beta``
    control the constant, lagged squared return and lagged variance terms,
    respectively.

    Args:
        returns: Series of asset returns.
        horizon: Number of future periods to forecast.
        omega: Constant term of the model.
        alpha: Coefficient for lagged squared returns.
        beta: Coefficient for lagged variance.

    Returns:
        Forecasted volatility values for horizons ``1`` to ``horizon``.
    """

    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    returns = returns.dropna().astype(float)
    if returns.empty:
        raise ValueError("returns series is empty")

    variance = float(returns.var())  # type: ignore[arg-type]
    prev_ret = float(returns.iloc[0])  # type: ignore[arg-type]
    for r in returns.iloc[1:]:
        variance = omega + alpha * prev_ret**2 + beta * variance
        prev_ret = float(r)  # type: ignore[arg-type]

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
    """Estimate Black-Scholes implied volatility.

    Args:
        price: Observed market price of the option.
        spot: Current underlying asset price.
        strike: Option strike price.
        time: Time to maturity in years.
        rate: Continuously compounded risk-free rate as a decimal.
        option_type: ``"call"`` for a call option or ``"put"`` for a put option.
        initial_vol: Starting guess for volatility.
        tol: Convergence tolerance for the iterative solver.
        max_iter: Maximum number of Newton iterations.

    Returns:
        Implied volatility that matches the observed price.

    Raises:
        ValueError: If ``option_type`` is not ``"call"`` or ``"put"``.
        RuntimeError: If the method fails to converge within ``max_iter`` steps.
    """

    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def norm_pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    volatility = initial_vol
    for _ in range(max_iter):
        d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * time) / (
            volatility * math.sqrt(time)
        )
        d2 = d1 - volatility * math.sqrt(time)

        if option_type == "call":
            model_price = spot * norm_cdf(d1) - strike * math.exp(
                -rate * time
            ) * norm_cdf(d2)
        else:
            model_price = strike * math.exp(-rate * time) * norm_cdf(
                -d2
            ) - spot * norm_cdf(-d1)

        vega = spot * math.sqrt(time) * norm_pdf(d1)

        diff = model_price - price
        if abs(diff) < tol:
            return volatility

        volatility -= diff / vega

    raise RuntimeError("implied volatility did not converge")


def compute_regime_probabilities(returns: pd.Series, n_iter: int = 10) -> pd.DataFrame:
    """Estimate volatility regimes using a two-state Markov model.

    The function fits a simple Gaussian hidden Markov model (HMM) with
    low- and high-volatility states. It iteratively updates regime
    probabilities using the expectation-maximization algorithm. Returns
    must represent asset returns and are assumed to have zero mean.

    Args:
        returns: Series of asset returns.
        n_iter: Number of EM iterations used for parameter estimation.

    Returns:
        DataFrame with columns ``low`` and ``high`` giving the posterior
        probability of each regime.

    Raises:
        ValueError: If ``returns`` is empty.
    """

    returns = returns.dropna().astype(float)
    if returns.empty:
        raise ValueError("returns series is empty")

    r = returns.to_numpy()
    n = len(r)

    def logsumexp(a: np.ndarray) -> float:
        a_max = np.max(a)
        return a_max + np.log(np.exp(a - a_max).sum())

    sigma_low = float(np.std(r)) * 0.5
    sigma_high = float(np.std(r)) * 2.0
    trans = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=float)
    pi = np.array([0.5, 0.5], dtype=float)

    for _ in range(n_iter):
        log_lik = np.vstack(
            [
                -0.5 * ((r / sigma_low) ** 2 + np.log(2 * np.pi * sigma_low**2)),
                -0.5 * ((r / sigma_high) ** 2 + np.log(2 * np.pi * sigma_high**2)),
            ]
        ).T

        log_alpha = np.zeros((n, 2))
        log_alpha[0] = np.log(pi) + log_lik[0]
        for t in range(1, n):
            for j in range(2):
                log_alpha[t, j] = log_lik[t, j] + logsumexp(
                    log_alpha[t - 1] + np.log(trans[:, j])
                )

        log_beta = np.zeros((n, 2))
        for t in range(n - 2, -1, -1):
            for i in range(2):
                log_beta[t, i] = logsumexp(
                    np.log(trans[i]) + log_lik[t + 1] + log_beta[t + 1]
                )

        log_gamma = log_alpha + log_beta
        norm = np.apply_along_axis(logsumexp, 1, log_gamma)
        gamma = np.exp(log_gamma - norm[:, None])

        log_xi = np.zeros((n - 1, 2, 2))
        for t in range(n - 1):
            for i in range(2):
                for j in range(2):
                    log_xi[t, i, j] = (
                        log_alpha[t, i]
                        + np.log(trans[i, j])
                        + log_lik[t + 1, j]
                        + log_beta[t + 1, j]
                    )
            log_xi[t] -= logsumexp(log_xi[t].ravel())
        xi = np.exp(log_xi)

        pi = gamma[0]
        trans = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, None]
        sigma_low = math.sqrt((gamma[:, 0] * r**2).sum() / gamma[:, 0].sum())
        sigma_high = math.sqrt((gamma[:, 1] * r**2).sum() / gamma[:, 1].sum())

    return pd.DataFrame(gamma, index=returns.index, columns=["low", "high"])
