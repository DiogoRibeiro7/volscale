# volnorm

Volnorm is a lightweight toolkit for volatility-normalized feature engineering in
financial time series. It contains helper functions for rolling statistics,
log-return computation and normalization utilities that make it easier to build
stable features.

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency
management and packaging. After cloning the repository install the
dependencies and the package with:

```bash
poetry install
```

## Running the tests

Tests rely on `pytest` and the development tools listed in
`pyproject.toml`. Run the suite from the repository root:

```bash
poetry run pytest -q
```

The CI also checks formatting, static types and linting. You can run them
manually using:

```bash
poetry run ruff check .
poetry run black --check .
poetry run mypy .
```

## Command line interface

Generate normalized features directly from CSV:

```bash
poetry run volnorm prices.csv --column close --window 20 --output features.csv
```

If `--output` is omitted the features are printed to standard output.

## Usage example

```python
import pandas as pd
from volnorm import build_normalized_features

# also available:
#   compute_true_range, compute_atr, compute_mad, classify_volatility

prices = pd.Series([100, 101, 102, 103, 104])
features = build_normalized_features(prices, window=3)
print(features.dropna())
```

## Project layout

```plaintext
src/
└── volnorm/
    ├── __init__.py
    ├── core.py           # basic transformations like log returns
    ├── rolling.py        # SMA, EMA, ATR proxy, std, etc.
    ├── volatility.py     # true range, ATR, MAD, classification
    ├── normalize.py      # volatility normalization
    └── features.py       # feature builder
tests/
    └── ...               # unit tests
```

Further planned work is described in [ROADMAP.md](ROADMAP.md).
