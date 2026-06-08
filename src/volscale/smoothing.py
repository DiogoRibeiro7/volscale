"""Compatibility wrapper for exploratory preprocessing helpers."""

from ._warnings import warn_legacy_wrapper

__all__ = ["low_pass_filter"]


def low_pass_filter(*args, **kwargs):
    """Compatibility wrapper for ``volscale.experimental.preprocessing.low_pass_filter``."""
    warn_legacy_wrapper()
    from .experimental.preprocessing import low_pass_filter as _low_pass_filter

    return _low_pass_filter(*args, **kwargs)
