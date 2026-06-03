from __future__ import annotations

import numpy as np
import pandas as pd

from .._validation import validate_finite_positive, validate_positive_int


def generate_synthetic_prices(
    length: int,
    *,
    seed: int | None = None,
    drift: float = 0.0,
    volatility: float = 0.01,
    start_price: float = 1.0,
    index: pd.Index | None = None,
) -> pd.Series:
    """Generate a synthetic price series using a log-normal random walk."""
    validate_positive_int(length, "length")
    validate_finite_positive(volatility, "volatility", allow_zero=True)
    validate_finite_positive(start_price, "start_price")
    rng = np.random.default_rng(seed)
    shocks = rng.normal(drift, volatility, size=length)
    prices = start_price * np.exp(np.cumsum(shocks))
    return pd.Series(prices, index=index)
