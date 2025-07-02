# VolNorm

VolNorm provides utilities for volatility-normalized feature engineering in financial time series. The library exposes rolling statistics, normalization helpers and a command line interface for quickly generating features from CSV data.

## Installation

Install the project with [Poetry](https://python-poetry.org/):

```bash
poetry install
```

## Quick start

Generate normalized features from a CSV file with the CLI:

```bash
poetry run volnorm prices.csv --column close --window 20 --output features.csv
```

Use the library directly in Python:

```python
import pandas as pd
from volnorm import build_normalized_features

prices = pd.Series([100, 101, 102, 103, 104])
features = build_normalized_features(prices, window=3)
print(features.dropna())
```

## Features

- Log returns and rolling volatility
- ATR proxies and true range calculations
- Volatility normalization of indicators
- Rolling statistics including SMA, EMA and WMA
- Realized volatility computed from intraday data
- GARCH-based volatility forecasts
- Implied volatility extraction from options prices
- Bollinger, Keltner and Donchian channel helpers
- Low-pass filtering and basic synthetic data generation
- Cross-sectional normalization across multiple assets
- Regime-switching models for volatility regimes
- Event-driven features for scheduled announcements

See [`notebooks/volnorm_example.ipynb`](notebooks/volnorm_example.ipynb) for a complete example.

## Project layout

```text
src/
└── volnorm/
    ├── core.py           # log returns and basic transforms
    ├── rolling.py        # moving averages and volatility metrics
    ├── volatility.py     # ATR, MAD and related indicators
    ├── smoothing.py      # denoising utilities
    ├── synthetic.py      # price series generators
    ├── normalize.py      # feature normalization helpers
    ├── features.py       # convenience feature builder
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
