import pandas as pd
import pytest
from volnorm._warnings import LEGACY_WRAPPER_WARNING
from volnorm.events import flag_event_window


def test_flag_event_window():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    events = pd.DatetimeIndex(["2024-01-03"])
    with pytest.deprecated_call(match=LEGACY_WRAPPER_WARNING):
        result = flag_event_window(idx, events, pre_event=1, post_event=1)
    expected = pd.Series([0.0, 1.0, 1.0, 1.0, 0.0], index=idx)
    pd.testing.assert_series_equal(result, expected)


def test_flag_event_window_supports_trading_days():
    idx = pd.bdate_range("2024-01-01", periods=5)
    events = pd.DatetimeIndex([idx[2]])
    with pytest.deprecated_call(match=LEGACY_WRAPPER_WARNING):
        result = flag_event_window(
            idx, events, pre_event=1, post_event=1, use_trading_days=True
        )
    expected = pd.Series([0.0, 1.0, 1.0, 1.0, 0.0], index=idx, dtype=float)
    pd.testing.assert_series_equal(result, expected)


def test_flag_event_window_rejects_negative_window():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    events = pd.DatetimeIndex(["2024-01-03"])
    with pytest.deprecated_call(match=LEGACY_WRAPPER_WARNING):
        with pytest.raises(ValueError, match="non-negative"):
            flag_event_window(idx, events, pre_event=-1)
