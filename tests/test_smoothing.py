import numpy as np
import pandas as pd
import pytest
from volscale._warnings import LEGACY_WRAPPER_WARNING
from volscale.smoothing import low_pass_filter


def test_low_pass_filter_basic():
    s = pd.Series([1, 2, 3, 4, 5])
    with pytest.deprecated_call(match=LEGACY_WRAPPER_WARNING):
        result = low_pass_filter(s, window=3, method="boxcar")
    expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0])
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_low_pass_filter_hann_shape():
    s = pd.Series(range(10))
    with pytest.deprecated_call(match=LEGACY_WRAPPER_WARNING):
        result = low_pass_filter(s, window=4)
    assert result.isna().sum() == 3


def test_low_pass_filter_rejects_unknown_method():
    s = pd.Series([1, 2, 3, 4])
    with pytest.deprecated_call(match=LEGACY_WRAPPER_WARNING):
        with pytest.raises(ValueError, match="method"):
            low_pass_filter(s, window=3, method="triangle")


def test_low_pass_filter_rejects_missing_values():
    s = pd.Series([1.0, np.nan, 3.0, 4.0])
    with pytest.deprecated_call(match=LEGACY_WRAPPER_WARNING):
        with pytest.raises(ValueError, match="missing data"):
            low_pass_filter(s, window=3)
