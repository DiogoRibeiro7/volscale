import pandas as pd
import pytest
from dataexcept import DataLoadingError

from volscale.datasets import load_spy_sample


def test_load_spy_sample():
    df = load_spy_sample()
    assert not df.empty
    assert set(df.columns) == {"Date", "Open", "High", "Low", "Close"}


def test_load_spy_sample_wraps_invalid_csv(monkeypatch):
    original = pd.errors.ParserError("corrupt sample")

    def fail(*args, **kwargs):
        raise original

    monkeypatch.setattr(pd, "read_csv", fail)
    with pytest.raises(DataLoadingError) as caught:
        load_spy_sample()

    assert caught.value.source == "volscale/data/spy_sample.csv"
    assert caught.value.original is original
    assert caught.value.__cause__ is original
