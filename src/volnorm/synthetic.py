"""Compatibility wrapper for exploratory synthetic-data helpers."""

from ._warnings import warn_legacy_wrapper

__all__ = ["generate_synthetic_prices"]


def generate_synthetic_prices(*args, **kwargs):
    """Compatibility wrapper for ``volnorm.experimental.synthetic.generate_synthetic_prices``."""
    warn_legacy_wrapper()
    from .experimental.synthetic import (
        generate_synthetic_prices as _generate_synthetic_prices,
    )

    return _generate_synthetic_prices(*args, **kwargs)
