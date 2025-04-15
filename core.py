import numpy as np
import pandas as pd

def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute log returns from a price series.
    """
    return np.log(prices / prices.shift(1))
  
