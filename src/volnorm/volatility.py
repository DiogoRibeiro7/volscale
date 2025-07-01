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

    # Compute log returns and drop the initial NaN introduced by the shift
    log_returns = np.log(prices / prices.shift(1)).dropna()

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

    returns = returns.dropna()
    if returns.empty:
        raise ValueError("returns series is empty")

    variance = returns.var()
    prev_ret = returns.iloc[0]
    for r in returns.iloc[1:]:
        variance = omega + alpha * prev_ret**2 + beta * variance
        prev_ret = r

    forecasts = []
    for _ in range(horizon):
        variance = omega + (alpha + beta) * variance
        forecasts.append(np.sqrt(variance))

    index = pd.RangeIndex(start=1, stop=horizon + 1, name="h")
    return pd.Series(forecasts, index=index, name="volatility")
