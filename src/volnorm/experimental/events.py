from __future__ import annotations

import pandas as pd


def flag_event_window(
    index: pd.DatetimeIndex,
    events: pd.DatetimeIndex,
    pre_event: int = 1,
    post_event: int = 1,
    *,
    use_trading_days: bool = False,
) -> pd.Series:
    """Flag scheduled event windows in a time index."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a pandas DatetimeIndex")
    if not isinstance(events, pd.DatetimeIndex):
        raise TypeError("events must be a pandas DatetimeIndex")
    if pre_event < 0 or post_event < 0:
        raise ValueError("pre_event and post_event must be non-negative")

    flags = pd.Series(0.0, index=index, dtype=float)
    if use_trading_days:
        positions = pd.Series(range(len(index)), index=index)
        for event in events:
            if event not in positions.index:
                continue
            event_pos = int(positions.loc[event])
            start_pos = max(event_pos - pre_event, 0)
            end_pos = min(event_pos + post_event + 1, len(index))
            flags.iloc[start_pos:end_pos] = 1.0
    else:
        for event in events:
            start = event - pd.Timedelta(days=pre_event)
            end = event + pd.Timedelta(days=post_event)
            mask = (index >= start) & (index <= end)
            flags.loc[mask] = 1.0
    return flags
