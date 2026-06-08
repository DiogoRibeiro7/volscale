# ROADMAP - Volatility Normalization Tools

This repository is focused on a small Python package for exploratory volatility normalization and feature engineering in financial time series. The goal is to keep the core utilities clean, validated, and reusable while clearly separating more experimental helpers from the stable API surface.

---

## Phase 1: Core Features

- [x] Compute log returns
- [x] Rolling standard deviation as volatility
- [x] Normalize features using N-day volatility
- [x] Simple Moving Average (SMA)
- [x] ATR proxy via rolling high-low range
- [x] Combine normalized features into a clean interface
- [x] Add unit tests for all core functions
- [x] Document all methods with type hints and docstrings

---

## Phase 2: Feature Expansion

- [x] Rolling metrics: EMA, max, min, z-score
- [x] Weighted Moving Average
- [x] True Range and proper ATR
- [x] Median Absolute Deviation as robust volatility
- [x] Volatility regime classification
- [x] Denoising and smoothing (e.g., low-pass filters)
- [x] Bollinger Bands indicator
- [x] Keltner Channels indicator
- [x] Donchian Channels indicator

---

## Phase 3: Package Readiness

- [x] Create `pyproject.toml` with Poetry or setuptools
- [x] Organize code into focused modules
- [x] Add CLI entry point for feature generation from CSV
- [x] Add examples in Jupyter notebooks
- [x] Make the test suite runnable from a clean checkout
- [x] Add input validation across public APIs
- [x] Split experimental helpers behind a dedicated module
- [x] Move exploratory implementations into a dedicated package subtree
- [x] Run tests, lint, format checks, and type checks in CI
- [x] Introduce typed configuration for the core feature pipeline
- [x] Add a dedicated release workflow with artifact builds
- [ ] Publish to PyPI (optional)

---

## Testing & Validation

- [x] Add `pytest`-based test suite
- [x] Include synthetic data generators for robustness testing
- [x] Add edge-case coverage for invalid inputs and numerical failure modes
- [x] Validate key numerical helpers against fixed reference benchmarks

---

## Name

The repository name is `volscale`.

---

## Vision

To offer a lightweight package for exploratory feature work on volatile financial data, with clear boundaries between stable utilities and experimental routines.

---

## Contributing

The project welcomes community involvement. If you have ideas for new features or improvements, open an issue or send a pull request. For questions, reach out to **Diogo Ribeiro** (`DiogoRibeiro7`) at
<diogo.debastos.ribeiro@gmail.com> (personal) or <dfr@esmad.ipp.pt> (professional).

## Maintainer

**Diogo Ribeiro** – ESMAD, Instituto Politécnico do Porto  
GitHub: [`DiogoRibeiro7`](https://github.com/DiogoRibeiro7)  
Personal: <diogo.debastos.ribeiro@gmail.com>  
Professional: <dfr@esmad.ipp.pt>  
ORCID: [0009-0001-2022-7072](https://orcid.org/0009-0001-2022-7072)

---

## Future Directions

Further ideas to extend volatility normalization:

- Better configuration objects for multi-feature pipelines
- Better reference validation for realized volatility
- Parameter estimation for GARCH-style models instead of fixed coefficients
- Safer solvers and stronger diagnostics for implied volatility
- Cross-sectional normalization across multiple assets *(implemented)*
- More robust regime models with convergence reporting
- Event-window helpers based on trading calendars
- Incremental computations for streaming data
- Visualization helpers for normalized indicators

