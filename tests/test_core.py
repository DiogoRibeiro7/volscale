import pandas as pd
import numpy as np
import pytest
from volnorm.core import compute_log_returns


def test_log_returns():
    prices = pd.Series([100, 105, 110])
    expected = np.log(prices / prices.shift(1))
    result = compute_log_returns(prices)
    pd.testing.assert_series_equal(result, expected)


def test_log_returns_rejects_non_positive_prices():
    prices = pd.Series([100, 0, 110])
    with pytest.raises(ValueError, match="strictly positive"):
        compute_log_returns(prices)
