import pandas as pd


def normalize_feature(feature: pd.Series, volatility: pd.Series) -> pd.Series:
    """Scale a feature by its corresponding volatility measure."""
    return feature / volatility
