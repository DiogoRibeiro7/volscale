import pandas as pd
from volnorm.synthetic import generate_synthetic_prices


def test_synthetic_length_and_seed():
    s1 = generate_synthetic_prices(10, seed=42)
    s2 = generate_synthetic_prices(10, seed=42)
    assert len(s1) == 10
    pd.testing.assert_series_equal(s1, s2)
