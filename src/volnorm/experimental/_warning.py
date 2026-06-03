from __future__ import annotations

import warnings


WARNING_MESSAGE = (
    "volnorm.experimental contains exploratory helpers with weaker statistical "
    "or numerical guarantees than the stable core API."
)


def warn_experimental() -> None:
    warnings.warn(WARNING_MESSAGE, category=UserWarning, stacklevel=2)
