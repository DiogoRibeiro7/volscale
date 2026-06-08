import pandas as pd
import pytest
from volscale.cli import main


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


def test_cli_can_keep_and_sort_index(tmp_path, capsys):
    df = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-01", "2024-01-02"],
            "close": [102, 100, 101],
        }
    )
    csv = tmp_path / "prices.csv"
    df.to_csv(csv, index=False)

    main(
        [
            str(csv),
            "--window",
            "2",
            "--date-column",
            "date",
            "--keep-index",
            "--sort-index",
        ]
    )
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("date,price_minus_sma_2_norm")
    assert "2024-01-02" in out


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


def test_cli_rejects_unknown_feature_name(tmp_path):
    df = pd.DataFrame({"close": [100, 101, 102, 103]})
    csv = tmp_path / "prices.csv"
    df.to_csv(csv, index=False)

    with pytest.raises(SystemExit, match="unknown feature names"):
        main([str(csv), "--window", "2", "--features", "banana"])


def test_cli_supports_json_output(tmp_path, capsys):
    df = pd.DataFrame({"close": [100, 101, 102, 103]})
    csv = tmp_path / "prices.csv"
    df.to_csv(csv, index=False)

    main([str(csv), "--window", "2", "--output-format", "json", "--dropna"])
    out = capsys.readouterr().out
    assert out.strip().startswith("[")
    assert '"log_return_norm_2"' in out


def test_cli_supports_custom_separators(tmp_path, capsys):
    df = pd.DataFrame({"close": [100, 101, 102, 103]})
    csv = tmp_path / "prices.csv"
    df.to_csv(csv, index=False, sep=";")

    main([str(csv), "--window", "2", "--input-sep", ";", "--output-sep", ";"])
    out = capsys.readouterr().out
    assert "price_minus_sma_2_norm;atr_proxy_2_norm;log_return_norm_2" in out


def test_cli_writes_output_file(tmp_path):
    df = pd.DataFrame({"close": [100, 101, 102, 103]})
    csv = tmp_path / "prices.csv"
    output = tmp_path / "nested" / "features.json"
    df.to_csv(csv, index=False)

    main(
        [
            str(csv),
            "--window",
            "2",
            "--output-format",
            "json",
            "--output",
            str(output),
            "--dropna",
        ]
    )
    assert output.exists()
    assert '"log_return_norm_2"' in output.read_text(encoding="utf-8")


def test_cli_rejects_empty_input_file(tmp_path):
    csv = tmp_path / "prices.csv"
    csv.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit, match="is empty"):
        main([str(csv), "--window", "2"])


def test_cli_rejects_invalid_dates(tmp_path):
    df = pd.DataFrame({"date": ["bad-date"], "close": [100]})
    csv = tmp_path / "prices.csv"
    df.to_csv(csv, index=False)

    with pytest.raises(SystemExit, match="Failed to parse date column"):
        main([str(csv), "--window", "2", "--date-column", "date"])
