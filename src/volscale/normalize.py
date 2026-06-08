import pandas as pd
import numpy as np
from ._validation import validate_series


def normalize_feature(feature: pd.Series, volatility: pd.Series) -> pd.Series:
    """Scale a feature by a volatility series, masking zero volatility."""
    feature = validate_series(feature, "feature")
    volatility = validate_series(volatility, "volatility")
    return feature.divide(volatility.replace(0.0, np.nan))


def compute_cross_sectional_zscore(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize values across assets using a z-score.

    This function standardizes each row of ``data`` independently so that the
    resulting values express the cross-sectional deviation from the mean on that
    particular date.

    Args:
        data: DataFrame where columns represent different assets and the index
            represents the date or timestamp.

    Returns:
        A ``DataFrame`` of the same shape containing the cross-sectional
        z-scores.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    mean = data.mean(axis=1)
    std = data.std(axis=1).replace(0.0, np.nan)
    return data.sub(mean, axis=0).div(std, axis=0)
