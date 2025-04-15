import pandas as pd
from volnorm.normalize import normalize_feature

def test_normalize_feature():
    f = pd.Series([10, 20, 30])
    v = pd.Series([2, 4, 5])
    result = normalize_feature(f, v)
    expected = pd.Series([5.0, 5.0, 6.0])
    pd.testing.assert_series_equal(result, expected)
