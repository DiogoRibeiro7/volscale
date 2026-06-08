import pandas as pd
import pytest
from volscale._warnings import LEGACY_WRAPPER_WARNING
from volscale.synthetic import generate_synthetic_prices


def test_synthetic_length_and_seed():
    with pytest.deprecated_call(match=LEGACY_WRAPPER_WARNING):
        s1 = generate_synthetic_prices(10, seed=42)
    with pytest.deprecated_call(match=LEGACY_WRAPPER_WARNING):
        s2 = generate_synthetic_prices(10, seed=42)
    assert len(s1) == 10
    pd.testing.assert_series_equal(s1, s2)


def test_synthetic_supports_custom_index_and_start_price():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    with pytest.deprecated_call(match=LEGACY_WRAPPER_WARNING):
        s = generate_synthetic_prices(3, seed=42, start_price=100.0, index=idx)
    assert s.index.equals(idx)
    assert s.iloc[0] > 0


def test_synthetic_rejects_bad_inputs():
    with pytest.deprecated_call(match=LEGACY_WRAPPER_WARNING):
        with pytest.raises(ValueError, match="positive integer"):
            generate_synthetic_prices(0)
    with pytest.deprecated_call(match=LEGACY_WRAPPER_WARNING):
        with pytest.raises(ValueError, match="strictly positive"):
            generate_synthetic_prices(3, start_price=0.0)
