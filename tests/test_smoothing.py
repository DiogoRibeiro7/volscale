import numpy as np
import pandas as pd
from volnorm.smoothing import low_pass_filter


def test_low_pass_filter_basic():
    s = pd.Series([1, 2, 3, 4, 5])
    result = low_pass_filter(s, window=3, method="boxcar")
    expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0])
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_low_pass_filter_hann_shape():
    s = pd.Series(range(10))
    result = low_pass_filter(s, window=4)
    assert result.isna().sum() == 3
