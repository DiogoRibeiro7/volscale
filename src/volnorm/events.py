"""Compatibility wrapper for exploratory event utilities."""

from ._warnings import warn_legacy_wrapper

__all__ = ["flag_event_window"]


def flag_event_window(*args, **kwargs):
    """Compatibility wrapper for ``volnorm.experimental.events.flag_event_window``."""
    warn_legacy_wrapper()
    from .experimental.events import flag_event_window as _flag_event_window

    return _flag_event_window(*args, **kwargs)
