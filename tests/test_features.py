import numpy as np
import pandas as pd
from volnorm.features import build_normalized_features

def test_build_normalized_features():
    np.random.seed(0)
    prices = pd.Series(np.cumprod(1 + np.random.normal(0, 0.01, 60)))
    df = build_normalized_features(prices, window=10)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 3  # sma, atr proxy, log return
    assert df.dropna().shape[0] > 0
