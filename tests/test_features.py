import numpy as np
import pandas as pd
import pytest
from volscale.features import (
    DEFAULT_FEATURES,
    SUPPORTED_FEATURES,
    FeatureConfig,
    build_normalized_features,
)


def test_build_normalized_features():
    np.random.seed(0)
    prices = pd.Series(np.cumprod(1 + np.random.normal(0, 0.01, 60)))
    df = build_normalized_features(prices, window=10)

    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == [
        "price_minus_sma_10_norm",
        "atr_proxy_10_norm",
        "log_return_norm_10",
    ]
    assert df.dropna().shape[0] > 0


def test_feature_defaults_are_supported():
    assert tuple(DEFAULT_FEATURES) == ("price_minus_sma", "atr_proxy", "log_return")
    assert set(DEFAULT_FEATURES) == SUPPORTED_FEATURES


def test_feature_config_defaults_are_valid():
    config = FeatureConfig(window=10)
    assert config.window == 10
    assert config.include == DEFAULT_FEATURES
    assert config.append_volatility is False


def test_build_normalized_features_rejects_bad_window():
    prices = pd.Series([100.0, 101.0, 102.0])
    with pytest.raises(ValueError, match="positive integer"):
        build_normalized_features(prices, window=0)


def test_build_normalized_features_rejects_non_positive_prices():
    prices = pd.Series([100.0, -101.0, 102.0])
    with pytest.raises(ValueError, match="strictly positive"):
        build_normalized_features(prices, window=2)


def test_build_normalized_features_supports_feature_selection_and_volatility():
    prices = pd.Series([100.0, 101.0, 103.0, 104.0, 105.0])
    config = FeatureConfig(
        window=2,
        include=("log_return", "price_minus_sma"),
        append_volatility=True,
    )
    df = build_normalized_features(prices, config=config)
    assert df.columns.tolist() == [
        "log_return_norm_2",
        "price_minus_sma_2_norm",
        "rolling_vol_2",
    ]


def test_build_normalized_features_rejects_unknown_features():
    prices = pd.Series([100.0, 101.0, 103.0, 104.0])
    with pytest.raises(ValueError, match="unknown feature names"):
        build_normalized_features(prices, window=2, include=["banana"])


def test_feature_config_rejects_unknown_features():
    with pytest.raises(ValueError, match="unknown feature names"):
        FeatureConfig(window=2, include=("banana",))


def test_build_normalized_features_rejects_mixed_config_and_kwargs():
    prices = pd.Series([100.0, 101.0, 103.0, 104.0])
    config = FeatureConfig(window=2)
    with pytest.raises(ValueError, match="cannot be combined"):
        build_normalized_features(prices, window=2, config=config)
