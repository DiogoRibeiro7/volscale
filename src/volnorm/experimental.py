"""Experimental APIs that are useful for exploration but not production-grade.

These helpers intentionally live behind a separate namespace because their
statistical assumptions, numerical behavior, or API maturity are weaker than the
stable core utilities exposed from ``volnorm`` directly.
"""

import warnings

from .events import flag_event_window as _flag_event_window
from .smoothing import low_pass_filter as _low_pass_filter
from .synthetic import generate_synthetic_prices as _generate_synthetic_prices
from .volatility import (
    compute_garch_forecast as _compute_garch_forecast,
    compute_implied_volatility as _compute_implied_volatility,
    compute_realized_volatility as _compute_realized_volatility,
    compute_regime_probabilities as _compute_regime_probabilities,
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

_WARNING = (
    "volnorm.experimental contains exploratory helpers with weaker statistical "
    "or numerical guarantees than the stable core API."
)


def _warn_experimental() -> None:
    warnings.warn(_WARNING, category=UserWarning, stacklevel=2)


def compute_realized_volatility(*args, **kwargs):
    """Experimental helper for realized volatility on intraday price series."""
    _warn_experimental()
    return _compute_realized_volatility(*args, **kwargs)


def compute_garch_forecast(*args, **kwargs):
    """Experimental fixed-parameter GARCH-style forecast helper."""
    _warn_experimental()
    return _compute_garch_forecast(*args, **kwargs)


def compute_implied_volatility(*args, **kwargs):
    """Experimental Black-Scholes implied volatility solver."""
    _warn_experimental()
    return _compute_implied_volatility(*args, **kwargs)


def compute_regime_probabilities(*args, **kwargs):
    """Experimental two-state regime probability estimator."""
    _warn_experimental()
    return _compute_regime_probabilities(*args, **kwargs)


def flag_event_window(*args, **kwargs):
    """Experimental event-window helper without exchange-calendar support."""
    _warn_experimental()
    return _flag_event_window(*args, **kwargs)


def low_pass_filter(*args, **kwargs):
    """Experimental smoothing helper for exploratory preprocessing."""
    _warn_experimental()
    return _low_pass_filter(*args, **kwargs)


def generate_synthetic_prices(*args, **kwargs):
    """Experimental synthetic price generator for toy simulations."""
    _warn_experimental()
    return _generate_synthetic_prices(*args, **kwargs)
