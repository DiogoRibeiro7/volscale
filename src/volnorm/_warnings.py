from __future__ import annotations

import warnings


LEGACY_WRAPPER_WARNING = (
    "This module is a compatibility wrapper around volnorm.experimental and may "
    "move or disappear in a future release. Prefer importing from "
    "volnorm.experimental directly."
)


def warn_legacy_wrapper() -> None:
    warnings.warn(LEGACY_WRAPPER_WARNING, category=DeprecationWarning, stacklevel=2)
