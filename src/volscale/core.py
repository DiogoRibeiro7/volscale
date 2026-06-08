import numpy as np
import pandas as pd
from ._validation import validate_strictly_positive_series


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from a price series."""
    prices = validate_strictly_positive_series(prices, "prices")
    return prices.div(prices.shift(1)).apply(np.log)
