import pandas as pd
import numpy as np
from volnorm.core import compute_log_returns

def test_log_returns():
    prices = pd.Series([100, 105, 110])
    expected = np.log(prices / prices.shift(1))
    result = compute_log_returns(prices)
    pd.testing.assert_series_equal(result, expected)
