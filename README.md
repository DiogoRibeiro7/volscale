# VolNorm

VolNorm provides a compact set of utilities for exploratory volatility-scaled feature engineering in financial time series. The library focuses on rolling statistics, normalization helpers, and a small command line interface for generating features from CSV data.

## Installation

Install the project with [Poetry](https://python-poetry.org/):

```bash
poetry install
```

For local verification without Poetry-managed commands, the repository CI runs:

```bash
python -m black --check src tests
python -m ruff check .
python -m mypy .
pytest -q
```

## Quick start

Generate normalized features from a CSV file with the CLI:

```bash
poetry run volnorm prices.csv --column close --window 20 --output features.csv
```

Use the library directly in Python:

```python
import pandas as pd
from volnorm import FeatureConfig, build_normalized_features

prices = pd.Series([100, 101, 102, 103, 104])
config = FeatureConfig(
    window=3,
    include=("price_minus_sma", "log_return"),
    append_volatility=True,
)
features = build_normalized_features(
    prices,
    config=config,
)
print(features.dropna())
```

## Features

- Log returns and rolling volatility
- ATR proxies and true range calculations
- Volatility normalization of indicators
- Rolling statistics including SMA, EMA and WMA
- Bollinger, Keltner and Donchian channel helpers
- Cross-sectional normalization across multiple assets

Experimental helpers live under the `volnorm.experimental` package:

- Realized volatility from intraday data
- Simple fixed-parameter GARCH-style forecasts
- Black-Scholes implied volatility solving
- Two-state regime probability estimation
- Low-pass filtering, synthetic price generation, and event-window flags

These experimental helpers emit a runtime warning and should be treated as
exploratory utilities rather than stable modeling components.

See [`notebooks/volnorm_example.ipynb`](notebooks/volnorm_example.ipynb) for a simple walkthrough.

## Project layout

```text
src/
└── volnorm/
    ├── core.py           # log returns and basic transforms
    ├── rolling.py        # moving averages and volatility metrics
    ├── volatility.py     # stable ATR, MAD and channel indicators
    ├── normalize.py      # feature normalization helpers
    ├── features.py       # convenience feature builder
    ├── experimental/     # exploratory helpers and toy model routines
    └── cli.py            # command line interface
```

The package also includes a small SPY price snippet for demos:

```python
from volnorm import load_spy_sample
spy = load_spy_sample()
```

## Contributing

Issues and pull requests are welcome. For questions, contact **Diogo Ribeiro** (<diogo.debastos.ribeiro@gmail.com> or <dfr@esmad.ipp.pt>).

Maintainer: [DiogoRibeiro7](https://github.com/DiogoRibeiro7) – ESMAD, Instituto Politécnico do Porto. ORCID: [0009-0001-2022-7072](https://orcid.org/0009-0001-2022-7072).

Further plans are detailed in [ROADMAP.md](ROADMAP.md).

## Release Process

Version releases are tag-driven and do not mutate the repository from CI.

1. Update `tool.poetry.version` in `pyproject.toml`.
2. Run the local verification commands.
3. Commit the version bump.
4. Create and push a tag in the form `vX.Y.Z`.
5. The `Release` workflow verifies that the tag matches `pyproject.toml`, builds the wheel and sdist, and publishes them to PyPI.

Publishing assumes PyPI trusted publishing is configured for this repository.
