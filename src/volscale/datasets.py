import importlib.resources
import pandas as pd
from dataexcept import DataLoadingError


def load_spy_sample() -> pd.DataFrame:
    """Load the included SPY sample dataset."""
    source = "volscale/data/spy_sample.csv"
    try:
        with importlib.resources.files(__package__).joinpath(
            "data/spy_sample.csv"
        ).open() as f:
            return pd.read_csv(f, parse_dates=["Date"])
    except (
        OSError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        ValueError,
    ) as exc:
        raise DataLoadingError(source, exc) from exc
