from __future__ import annotations

import numpy as np
import pandas as pd

from .._validation import validate_positive_int, validate_series


def low_pass_filter(
    values: pd.Series, window: int, *, method: str = "hann"
) -> pd.Series:
    """Smooth a series using a simple low-pass filter."""
    values = validate_series(values, "values")
    validate_positive_int(window, "window")
    if window <= 1:
        return values

    if method == "hann":
        weights = np.hanning(window)
    elif method == "boxcar":
        weights = np.ones(window)
    else:
        raise ValueError("method must be either 'hann' or 'boxcar'")
    if values.isna().any():
        raise ValueError("values must not contain missing data")

    weights = weights / weights.sum()
    filtered = np.convolve(values, weights, mode="valid")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    result.iloc[window - 1 :] = pd.Series(filtered, index=result.index[window - 1 :])
    return result
