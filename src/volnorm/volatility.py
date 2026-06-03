import numpy as np
import pandas as pd
from ._validation import (
    validate_finite_positive,
    validate_positive_int,
    validate_series,
)
from .rolling import compute_sma, compute_rolling_std


def compute_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Return the True Range for each period."""
    high = validate_series(high, "high")
    low = validate_series(low, "low")
    close = validate_series(close, "close")
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
    validate_positive_int(window, "window")
    tr = compute_true_range(high, low, close)
    if method == "ema":
        return tr.ewm(span=window, adjust=False).mean()
    if method == "sma":
        return tr.rolling(window=window, min_periods=window).mean()
    raise ValueError("method must be either 'ema' or 'sma'")


def compute_mad(values: pd.Series, window: int) -> pd.Series:
    """Rolling Median Absolute Deviation."""
    values = validate_series(values, "values")
    validate_positive_int(window, "window")
    return values.rolling(window=window, min_periods=window).apply(
        lambda x: np.median(np.abs(x - np.median(x)))
    )


def classify_volatility(
    volatility: pd.Series,
    low_quantile: float = 0.25,
    high_quantile: float = 0.75,
) -> pd.Series:
    """Classify volatility levels using quantile thresholds."""
    volatility = validate_series(volatility, "volatility")
    if not 0 <= low_quantile <= 1 or not 0 <= high_quantile <= 1:
        raise ValueError("quantiles must be between 0 and 1")
    if low_quantile >= high_quantile:
        raise ValueError("low_quantile must be smaller than high_quantile")
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
    validate_finite_positive(num_std, "num_std", allow_zero=True)
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
    validate_finite_positive(atr_multiplier, "atr_multiplier", allow_zero=True)
    ema = close.ewm(span=window, adjust=False).mean()
    atr = compute_atr(high, low, close, window)
    upper = ema + atr_multiplier * atr
    lower = ema - atr_multiplier * atr
    return pd.DataFrame({"middle": ema, "upper": upper, "lower": lower})


def compute_donchian_channels(
    high: pd.Series, low: pd.Series, window: int
) -> pd.DataFrame:
    """Return Donchian Channels using rolling extremes."""
    high = validate_series(high, "high")
    low = validate_series(low, "low")
    validate_positive_int(window, "window")
    upper = high.rolling(window=window, min_periods=window).max()
    lower = low.rolling(window=window, min_periods=window).min()
    middle = (upper + lower) / 2
    return pd.DataFrame({"middle": middle, "upper": upper, "lower": lower})
