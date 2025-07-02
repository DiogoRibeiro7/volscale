import pandas as pd
from volnorm.events import flag_event_window


def test_flag_event_window():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    events = pd.DatetimeIndex(["2024-01-03"])
    result = flag_event_window(idx, events, pre_event=1, post_event=1)
    expected = pd.Series([0.0, 1.0, 1.0, 1.0, 0.0], index=idx)
    pd.testing.assert_series_equal(result, expected)
