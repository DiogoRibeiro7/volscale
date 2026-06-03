import pandas as pd
import pytest
from volnorm.cli import main


def test_cli_basic(tmp_path, capsys):
    df = pd.DataFrame({"close": [100, 101, 102]})
    csv = tmp_path / "prices.csv"
    df.to_csv(csv, index=False)

    main([str(csv), "--window", "2"])
    out = capsys.readouterr().out
    assert "price_minus_sma_2_norm" in out


def test_cli_rejects_missing_column(tmp_path):
    df = pd.DataFrame({"price": [100, 101, 102]})
    csv = tmp_path / "prices.csv"
    df.to_csv(csv, index=False)

    with pytest.raises(SystemExit, match="Column 'close' not found"):
        main([str(csv), "--window", "2"])


def test_cli_supports_date_column(tmp_path, capsys):
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "close": [100, 101, 102],
        }
    )
    csv = tmp_path / "prices.csv"
    df.to_csv(csv, index=False)

    main([str(csv), "--window", "2", "--date-column", "date"])
    out = capsys.readouterr().out
    assert "log_return_norm_2" in out


def test_cli_supports_feature_selection_and_volatility(tmp_path, capsys):
    df = pd.DataFrame({"close": [100, 101, 102, 103]})
    csv = tmp_path / "prices.csv"
    df.to_csv(csv, index=False)

    main(
        [
            str(csv),
            "--window",
            "2",
            "--features",
            "log_return,atr_proxy",
            "--include-volatility",
        ]
    )
    out = capsys.readouterr().out
    assert "log_return_norm_2" in out
    assert "atr_proxy_2_norm" in out
    assert "rolling_vol_2" in out
