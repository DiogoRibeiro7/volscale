import numpy as np
import pandas as pd
import pytest
from volnorm.features import build_normalized_features


def test_build_normalized_features():
    np.random.seed(0)
    prices = pd.Series(np.cumprod(1 + np.random.normal(0, 0.01, 60)))
    df = build_normalized_features(prices, window=10)

    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == [
        "sma_10_norm",
        "atr_proxy_10_norm",
        "log_return_norm_10",
    ]
    assert df.dropna().shape[0] > 0


def test_build_normalized_features_rejects_bad_window():
    prices = pd.Series([100.0, 101.0, 102.0])
    with pytest.raises(ValueError, match="positive integer"):
        build_normalized_features(prices, window=0)


def test_build_normalized_features_rejects_non_positive_prices():
    prices = pd.Series([100.0, -101.0, 102.0])
    with pytest.raises(ValueError, match="strictly positive"):
        build_normalized_features(prices, window=2)
