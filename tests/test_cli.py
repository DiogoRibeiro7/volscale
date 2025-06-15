import pandas as pd
from volnorm.cli import main


def test_cli_basic(tmp_path, capsys):
    df = pd.DataFrame({"close": [100, 101, 102]})
    csv = tmp_path / "prices.csv"
    df.to_csv(csv, index=False)

    main([str(csv), "--window", "2"])
    out = capsys.readouterr().out
    assert "sma_2_norm" in out
