import numpy as np
import pandas as pd


def generate_synthetic_prices(
    length: int,
    *,
    seed: int | None = None,
    drift: float = 0.0,
    volatility: float = 0.01,
) -> pd.Series:
    """Generate a synthetic price series using a log-normal random walk."""
    if seed is not None:
        np.random.seed(seed)
    # Start at 1.0 for simplicity
    shocks = np.random.normal(drift, volatility, size=length)
    prices = np.exp(np.cumsum(shocks))
    return pd.Series(prices)
