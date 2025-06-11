import pandas as pd

def compute_ema(prices: pd.Series, span: int) -> pd.Series:
    """Compute the exponential moving average."""
    return prices.ewm(span=span, adjust=False).mean()

def compute_rolling_std(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=window, min_periods=window).std()

def compute_sma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window=window, min_periods=window).mean()

def compute_atr_proxy(prices: pd.Series, window: int) -> pd.Series:
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
