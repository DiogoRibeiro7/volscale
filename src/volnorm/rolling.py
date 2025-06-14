import numpy as np
import pandas as pd


def compute_ema(prices: pd.Series, span: int) -> pd.Series:
    """Compute the exponential moving average."""
    return prices.ewm(span=span, adjust=False).mean()


def compute_rolling_std(values: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation with full window requirement."""
    return values.rolling(window=window, min_periods=window).std()


def compute_sma(prices: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return prices.rolling(window=window, min_periods=window).mean()


def compute_atr_proxy(prices: pd.Series, window: int) -> pd.Series:
    """Approximate ATR using the rolling high-low range."""
    high = prices.rolling(window=window, min_periods=window).max()
    low = prices.rolling(window=window, min_periods=window).min()
    return high - low


def compute_rolling_max(values: pd.Series, window: int) -> pd.Series:
    """Rolling maximum."""
    return values.rolling(window=window, min_periods=window).max()


def compute_rolling_min(values: pd.Series, window: int) -> pd.Series:
    """Rolling minimum."""
    return values.rolling(window=window, min_periods=window).min()


def compute_zscore(values: pd.Series, window: int) -> pd.Series:
    """Rolling z-score based on rolling mean and std."""
    mean = compute_sma(values, window)
    std = compute_rolling_std(values, window)
    return (values - mean) / std


def compute_wma(values: pd.Series, window: int) -> pd.Series:
    """Weighted moving average with linearly increasing weights."""
    weights = np.arange(1, window + 1)
    return values.rolling(window=window, min_periods=window).apply(
        lambda x: np.dot(x, weights) / weights.sum(),
        raw=True,
    )
