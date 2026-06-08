from dataclasses import dataclass
from collections.abc import Iterable

import pandas as pd

from ._validation import validate_positive_int, validate_strictly_positive_series
from .core import compute_log_returns
from .rolling import compute_rolling_std, compute_sma, compute_atr_proxy
from .normalize import normalize_feature

DEFAULT_FEATURES = ("price_minus_sma", "atr_proxy", "log_return")
SUPPORTED_FEATURES = frozenset(DEFAULT_FEATURES)


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for the normalized feature pipeline."""

    window: int
    include: tuple[str, ...] = DEFAULT_FEATURES
    append_volatility: bool = False

    def __post_init__(self) -> None:
        validate_positive_int(self.window, "window")
        unknown = sorted(set(self.include) - SUPPORTED_FEATURES)
        if unknown:
            allowed = ", ".join(sorted(SUPPORTED_FEATURES))
            bad = ", ".join(unknown)
            raise ValueError(
                f"unknown feature names: {bad}. Supported features: {allowed}"
            )

    @classmethod
    def from_inputs(
        cls,
        *,
        window: int,
        include: Iterable[str] | None = None,
        append_volatility: bool = False,
    ) -> "FeatureConfig":
        selected = tuple(include) if include is not None else DEFAULT_FEATURES
        return cls(
            window=window,
            include=selected,
            append_volatility=append_volatility,
        )


def build_normalized_features(
    prices: pd.Series,
    window: int | None = None,
    *,
    config: FeatureConfig | None = None,
    include: Iterable[str] | None = None,
    append_volatility: bool = False,
) -> pd.DataFrame:
    """Return volatility-scaled exploratory features from a validated config.

    The default output intentionally stays small and focuses on quantities with
    clearer semantics than raw price-level indicators. In particular, the moving
    average feature is represented as the gap between price and SMA, scaled by
    rolling return volatility.
    """
    prices = validate_strictly_positive_series(prices, "prices")
    if config is None:
        if window is None:
            raise ValueError("window is required when config is not provided")
        resolved = FeatureConfig.from_inputs(
            window=window,
            include=include,
            append_volatility=append_volatility,
        )
    else:
        if window is not None or include is not None or append_volatility:
            raise ValueError(
                "config cannot be combined with window, include, or append_volatility"
            )
        resolved = config

    log_returns = compute_log_returns(prices)
    volatility = compute_rolling_std(log_returns, resolved.window)

    sma = compute_sma(prices, resolved.window)
    atr_proxy = compute_atr_proxy(prices, resolved.window)
    feature_map = {
        f"price_minus_sma_{resolved.window}_norm": normalize_feature(
            prices - sma, volatility
        ),
        f"atr_proxy_{resolved.window}_norm": normalize_feature(atr_proxy, volatility),
        f"log_return_norm_{resolved.window}": normalize_feature(
            log_returns, volatility
        ),
    }
    name_map = {
        "price_minus_sma": f"price_minus_sma_{resolved.window}_norm",
        "atr_proxy": f"atr_proxy_{resolved.window}_norm",
        "log_return": f"log_return_norm_{resolved.window}",
    }

    data = {name_map[name]: feature_map[name_map[name]] for name in resolved.include}
    if resolved.append_volatility:
        data[f"rolling_vol_{resolved.window}"] = volatility
    return pd.DataFrame(data)
