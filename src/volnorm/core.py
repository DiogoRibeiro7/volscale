import numpy as np
import pandas as pd
from ._validation import validate_strictly_positive_series


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from a price series."""
    prices = validate_strictly_positive_series(prices, "prices")
    return np.log(prices / prices.shift(1))
