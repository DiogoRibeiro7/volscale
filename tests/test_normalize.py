import pandas as pd
import numpy as np
import pytest
from volscale.normalize import normalize_feature, compute_cross_sectional_zscore


def test_normalize_feature():
    f = pd.Series([10, 20, 30])
    v = pd.Series([2, 4, 5])
    result = normalize_feature(f, v)
    expected = pd.Series([5.0, 5.0, 6.0])
    pd.testing.assert_series_equal(result, expected)


def test_compute_cross_sectional_zscore():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 2, 4], "c": [3, 2, 5]})
    z = compute_cross_sectional_zscore(df)
    expected = (df.sub(df.mean(axis=1), axis=0)).div(df.std(axis=1), axis=0)
    pd.testing.assert_frame_equal(z, expected)


def test_normalize_feature_masks_zero_volatility():
    f = pd.Series([10.0, 20.0])
    v = pd.Series([0.0, 4.0])
    result = normalize_feature(f, v)
    expected = pd.Series([np.nan, 5.0])
    pd.testing.assert_series_equal(result, expected)


def test_compute_cross_sectional_zscore_requires_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        compute_cross_sectional_zscore(pd.Series([1, 2, 3]))  # type: ignore[arg-type]
