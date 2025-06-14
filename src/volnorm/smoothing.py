import numpy as np
import pandas as pd


def low_pass_filter(
    values: pd.Series, window: int, *, method: str = "hann"
) -> pd.Series:
    """Smooth a series using a simple low-pass filter."""
    if window <= 1:
        return values

    if method == "hann":
        weights = np.hanning(window)
    else:
        weights = np.ones(window)
    weights = weights / weights.sum()

    filtered = np.convolve(values, weights, mode="valid")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    result.iloc[window - 1 :] = pd.Series(filtered, index=result.index[window - 1 :])
    return result
