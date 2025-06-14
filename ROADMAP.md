# 📈 ROADMAP – Volatility Normalization Tools

This repository is dedicated to building a pure Python toolkit for volatility normalization and feature engineering in financial time series. The goal is to offer clean, robust, and reusable utilities that improve modeling stability and performance—without relying on third-party finance libraries.

---

## ✅ Phase 1: Core Features (MVP)

- [x] Compute log returns
- [x] Rolling standard deviation as volatility
- [x] Normalize features using N-day volatility
- [x] Simple Moving Average (SMA)
- [x] ATR proxy via rolling high-low range
- [x] Combine normalized features into a clean interface
- [x] Add unit tests for all core functions
- [x] Document all methods with type hints and docstrings

---

## 🚧 Phase 2: Feature Expansion

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

## 🚀 Phase 3: Package Readiness

- [x] Create `pyproject.toml` with Poetry or setuptools
- [x] Organize modules into `core/`, `features/`, `transforms/`
- [x] Add CLI entry point for feature generation from CSV
- [x] Add examples in Jupyter notebooks
- [ ] Publish to PyPI (optional)

---

## 🧪 Testing & Validation

- [x] Add `pytest`-based test suite
- [x] Include synthetic data generators for robustness testing
- [x] Validate against known data (e.g., SPY, AAPL)

---

## 💡 Name Suggestions

The current shortlist for the repository name includes:

- `volnorm` – Volatility normalization, short and to the point  
- `voltools` – Toolbox for volatility-based transformations  
- `featvol` – Feature engineering through volatility  
- `volnify` – Normalize using volatility (catchy)  
- `bounded-features` – Emphasizes feature stabilization  
- `voltkit` – A volatility feature engineering kit  
- `volsignal` – Turning volatility into signal  

**Recommended:** `volnorm` for clarity and naming consistency.

---

## ✨ Vision

To offer a lightweight, dependency-free package that helps researchers, quants, and data scientists make their features bounded, stable, and more model-friendly—especially when working with volatile financial data.

