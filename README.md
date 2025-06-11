# volnorm

Volnorm is a lightweight toolkit for volatility-normalized feature engineering in
financial time series. It contains helper functions for rolling statistics,
log-return computation and normalization utilities that make it easier to build
stable features.

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency
management. After cloning the repository install the dependencies with:

```bash
poetry install --no-root
```

## Running the tests

Tests rely on `pytest` and the development tools listed in
`pyproject.toml`. Run the suite from the repository root:

```bash
PYTHONPATH="$(pwd)/.." poetry run pytest -q
```

The CI also checks formatting, static types and linting. You can run them
manually using:

```bash
poetry run ruff volnorm tests
poetry run black --check volnorm tests
poetry run mypy volnorm
```

## Usage example

```python
import pandas as pd
from volnorm import build_normalized_features

prices = pd.Series([100, 101, 102, 103, 104])
features = build_normalized_features(prices, window=3)
print(features.dropna())
```

## Project layout

```plaintext
volnorm/
├── __init__.py
├── core.py           # basic transformations like log returns
├── rolling.py        # SMA, EMA, ATR proxy, std, etc.
├── normalize.py      # volatility normalization
├── features.py       # feature builder
└── tests/            # unit tests
```

Further planned work is described in [ROADMAP.md](ROADMAP.md).
