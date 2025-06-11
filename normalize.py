import pandas as pd


def normalize_feature(feature: pd.Series, volatility: pd.Series) -> pd.Series:
    return feature / volatility
