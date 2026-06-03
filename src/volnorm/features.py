from collections.abc import Iterable
import pandas as pd
from ._validation import validate_positive_int, validate_strictly_positive_series
from .core import compute_log_returns
from .rolling import compute_rolling_std, compute_sma, compute_atr_proxy
from .normalize import normalize_feature

DEFAULT_FEATURES = ("price_minus_sma", "atr_proxy", "log_return")
SUPPORTED_FEATURES = frozenset(DEFAULT_FEATURES)


def build_normalized_features(
    prices: pd.Series,
    window: int,
    *,
    include: Iterable[str] | None = None,
    append_volatility: bool = False,
) -> pd.DataFrame:
    """Return configurable volatility-scaled exploratory features.

    The default output intentionally stays small and focuses on quantities with
    clearer semantics than raw price-level indicators. In particular, the moving
    average feature is represented as the gap between price and SMA, scaled by
    rolling return volatility.
    """
    prices = validate_strictly_positive_series(prices, "prices")
    validate_positive_int(window, "window")
    selected = tuple(include) if include is not None else DEFAULT_FEATURES
    unknown = sorted(set(selected) - SUPPORTED_FEATURES)
    if unknown:
        allowed = ", ".join(sorted(SUPPORTED_FEATURES))
        bad = ", ".join(unknown)
        raise ValueError(f"unknown feature names: {bad}. Supported features: {allowed}")

    log_returns = compute_log_returns(prices)
    volatility = compute_rolling_std(log_returns, window)

    sma = compute_sma(prices, window)
    atr_proxy = compute_atr_proxy(prices, window)
    feature_map = {
        f"price_minus_sma_{window}_norm": normalize_feature(prices - sma, volatility),
        f"atr_proxy_{window}_norm": normalize_feature(atr_proxy, volatility),
        f"log_return_norm_{window}": normalize_feature(log_returns, volatility),
    }
    name_map = {
        "price_minus_sma": f"price_minus_sma_{window}_norm",
        "atr_proxy": f"atr_proxy_{window}_norm",
        "log_return": f"log_return_norm_{window}",
    }

    data = {name_map[name]: feature_map[name_map[name]] for name in selected}
    if append_volatility:
        data[f"rolling_vol_{window}"] = volatility
    return pd.DataFrame(data)
