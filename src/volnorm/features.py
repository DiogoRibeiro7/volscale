import pandas as pd
from ._validation import validate_positive_int, validate_strictly_positive_series
from .core import compute_log_returns
from .rolling import compute_rolling_std, compute_sma, compute_atr_proxy
from .normalize import normalize_feature


def build_normalized_features(prices: pd.Series, window: int) -> pd.DataFrame:
    """Return a small set of volatility-scaled exploratory features."""
    prices = validate_strictly_positive_series(prices, "prices")
    validate_positive_int(window, "window")

    log_returns = compute_log_returns(prices)
    volatility = compute_rolling_std(log_returns, window)

    sma = compute_sma(prices, window)
    atr_proxy = compute_atr_proxy(prices, window)

    return pd.DataFrame(
        {
            f"sma_{window}_norm": normalize_feature(sma, volatility),
            f"atr_proxy_{window}_norm": normalize_feature(atr_proxy, volatility),
            f"log_return_norm_{window}": normalize_feature(log_returns, volatility),
        }
    )
