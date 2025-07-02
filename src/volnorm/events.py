import pandas as pd


def flag_event_window(
    index: pd.DatetimeIndex,
    events: pd.DatetimeIndex,
    pre_event: int = 1,
    post_event: int = 1,
) -> pd.Series:
    """Flag scheduled event windows in a time index.

    Args:
        index: Date index for which the feature is computed.
        events: Dates of scheduled events.
        pre_event: Number of days before an event considered part of the window.
        post_event: Number of days after an event considered part of the window.

    Returns:
        Series with ``1.0`` for rows falling inside an event window and ``0.0`` otherwise.
    """
    flags = pd.Series(0.0, index=index)
    for event in events:
        start = event - pd.Timedelta(days=pre_event)
        end = event + pd.Timedelta(days=post_event)
        mask = (index >= start) & (index <= end)
        flags.loc[mask] = 1.0
    return flags
