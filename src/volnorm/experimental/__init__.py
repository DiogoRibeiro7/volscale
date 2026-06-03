"""Exploratory helpers with weaker guarantees than the stable core API."""

from __future__ import annotations

from ._warning import warn_experimental
from .events import flag_event_window as _flag_event_window
from .preprocessing import low_pass_filter as _low_pass_filter
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


def compute_realized_volatility(*args, **kwargs):
    """Experimental helper for realized volatility on intraday price series."""
    warn_experimental()
    return _compute_realized_volatility(*args, **kwargs)


def compute_garch_forecast(*args, **kwargs):
    """Experimental fixed-parameter GARCH-style forecast helper."""
    warn_experimental()
    return _compute_garch_forecast(*args, **kwargs)


def compute_implied_volatility(*args, **kwargs):
    """Experimental Black-Scholes implied volatility solver."""
    warn_experimental()
    return _compute_implied_volatility(*args, **kwargs)


def compute_regime_probabilities(*args, **kwargs):
    """Experimental two-state regime probability estimator."""
    warn_experimental()
    return _compute_regime_probabilities(*args, **kwargs)


def flag_event_window(*args, **kwargs):
    """Experimental event-window helper without exchange-calendar support."""
    warn_experimental()
    return _flag_event_window(*args, **kwargs)


def low_pass_filter(*args, **kwargs):
    """Experimental smoothing helper for exploratory preprocessing."""
    warn_experimental()
    return _low_pass_filter(*args, **kwargs)


def generate_synthetic_prices(*args, **kwargs):
    """Experimental synthetic price generator for toy simulations."""
    warn_experimental()
    return _generate_synthetic_prices(*args, **kwargs)
