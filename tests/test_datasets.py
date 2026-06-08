from volscale.datasets import load_spy_sample


def test_load_spy_sample():
    df = load_spy_sample()
    assert not df.empty
    assert set(df.columns) == {"Date", "Open", "High", "Low", "Close"}
