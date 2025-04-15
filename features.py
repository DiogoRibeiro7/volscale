import numpy as np
import pandas as pd
from typing import Tuple

def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute log returns from price series.
    
    Parameters:
        prices (pd.Series): Series of prices indexed by datetime.
    
    Returns:
        pd.Series: Log returns.
    """
    returns = np.log(prices / prices.shift(1))
    return returns


def compute_rolling_std(values: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling standard deviation (used as volatility).
    
    Parameters:
        values (pd.Series): Input series (e.g., log returns).
        window (int): Window length for rolling computation.
    
    Returns:
        pd.Series: Rolling standard deviation.
    """
    return values.rolling(window=window, min_periods=window).std()


def compute_sma(prices: pd.Series, window: int) -> pd.Series:
    """
    Compute simple moving average (SMA).
    
    Parameters:
        prices (pd.Series): Series of prices.
        window (int): Window size.
    
    Returns:
        pd.Series: SMA series.
    """
    return prices.rolling(window=window, min_periods=window).mean()


def compute_atr_proxy(prices: pd.Series, window: int) -> pd.Series:
    """
    Compute a simple proxy for ATR using high-low range over a window.
    
    Parameters:
        prices (pd.Series): Series of prices.
        window (int): Window size.
    
    Returns:
        pd.Series: ATR proxy series.
    """
    high = prices.rolling(window=window, min_periods=window).max()
    low = prices.rolling(window=window, min_periods=window).min()
    return high - low


def normalize_feature(feature: pd.Series, volatility: pd.Series) -> pd.Series:
    """
    Normalize a feature by volatility.
    
    Parameters:
        feature (pd.Series): Input feature.
        volatility (pd.Series): Volatility series.
    
    Returns:
        pd.Series: Normalized feature.
    """
    return feature / volatility


def build_normalized_features(prices: pd.Series, window: int) -> pd.DataFrame:
    """
    Construct normalized features for a financial time series.

    Includes:
        - Normalized SMA
        - Normalized ATR proxy
        - Normalized log return

    Parameters:
        prices (pd.Series): Series of prices.
        window (int): Rolling window for both features and volatility.

    Returns:
        pd.DataFrame: Normalized features.
    """
    log_returns = compute_log_returns(prices)
    volatility = compute_rolling_std(log_returns, window)

    sma = compute_sma(prices, window)
    atr_proxy = compute_atr_proxy(prices, window)

    return pd.DataFrame({
        f"sma_{window}_norm": normalize_feature(sma, volatility),
        f"atr_proxy_{window}_norm": normalize_feature(atr_proxy, volatility),
        f"log_return_norm_{window}": normalize_feature(log_returns, volatility)
    })


# Example usage (for testing or notebook):
if __name__ == "__main__":
    np.random.seed(0)
    idx = pd.date_range("2022-01-01", periods=250)
    simulated_prices = pd.Series(np.cumprod(1 + np.random.normal(0, 0.01, 250)), index=idx)

    features = build_normalized_features(simulated_prices, window=30)
    print(features.dropna().head())
