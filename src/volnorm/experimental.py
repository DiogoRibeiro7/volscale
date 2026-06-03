"""Experimental APIs with lighter validation and weaker statistical guarantees."""

from .events import flag_event_window
from .smoothing import low_pass_filter
from .synthetic import generate_synthetic_prices
from .volatility import (
    compute_garch_forecast,
    compute_implied_volatility,
    compute_realized_volatility,
    compute_regime_probabilities,
)

__all__ = [
    "compute_realized_volatility",
    "compute_garch_forecast",
    "compute_implied_volatility",
    "compute_regime_probabilities",
    "flag_event_window",
    "low_pass_filter",
    "generate_synthetic_prices",
]
