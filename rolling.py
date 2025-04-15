import pandas as pd

def compute_rolling_std(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=window, min_periods=window).std()

def compute_sma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window=window, min_periods=window).mean()

def compute_atr_proxy(prices: pd.Series, window: int) -> pd.Series:
    high = prices.rolling(window=window, min_periods=window).max()
    low = prices.rolling(window=window, min_periods=window).min()
    return high - low
