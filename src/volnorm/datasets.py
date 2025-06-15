import importlib.resources
import pandas as pd


def load_spy_sample() -> pd.DataFrame:
    """Load the included SPY sample dataset."""
    with importlib.resources.files(__package__).joinpath(
        "data/spy_sample.csv"
    ).open() as f:
        return pd.read_csv(f, parse_dates=["Date"])
