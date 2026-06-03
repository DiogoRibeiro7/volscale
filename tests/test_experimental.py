import warnings

import pandas as pd

from volnorm import experimental
from volnorm.experimental._warning import WARNING_MESSAGE


def test_experimental_helpers_emit_warning():
    idx = pd.DatetimeIndex(
        [
            "2024-01-01 09:30",
            "2024-01-01 09:31",
            "2024-01-01 09:32",
        ]
    )
    prices = pd.Series([100.0, 101.0, 102.0], index=idx)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        experimental.compute_realized_volatility(prices)

    assert caught
    assert WARNING_MESSAGE in str(caught[0].message)
