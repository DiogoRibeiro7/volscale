from __future__ import annotations

import math

import pandas as pd


def validate_positive_int(value: int, name: str) -> int:
    """Validate that a parameter is a positive integer."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_series(prices: pd.Series, name: str) -> pd.Series:
    """Ensure a value is a pandas Series with numeric dtype."""
    if not isinstance(prices, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    return prices.astype(float)


def validate_strictly_positive_series(values: pd.Series, name: str) -> pd.Series:
    """Ensure a series is numeric and strictly positive after null filtering."""
    series = validate_series(values, name)
    non_null = series.dropna()
    if not non_null.empty and (non_null <= 0).any():
        raise ValueError(f"{name} must contain only strictly positive values")
    return series


def validate_finite_positive(
    value: float, name: str, *, allow_zero: bool = False
) -> float:
    """Validate a scalar numeric input."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    elif value <= 0:
        raise ValueError(f"{name} must be strictly positive")
    return value
