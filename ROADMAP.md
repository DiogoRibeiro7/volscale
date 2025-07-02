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

---

## 🤝 Contributing

The project welcomes community involvement. If you have ideas for new features or improvements, open an issue or send a pull request. For questions, reach out to **Diogo Ribeiro** (`DiogoRibeiro7`) at
<diogo.debastos.ribeiro@gmail.com> (personal) or <dfr@esmad.ipp.pt> (professional).

## 📫 Maintainer

**Diogo Ribeiro** – ESMAD, Instituto Politécnico do Porto  
GitHub: [`DiogoRibeiro7`](https://github.com/DiogoRibeiro7)  
Personal: <diogo.debastos.ribeiro@gmail.com>  
Professional: <dfr@esmad.ipp.pt>  
ORCID: [0009-0001-2022-7072](https://orcid.org/0009-0001-2022-7072)

---

## 🔮 Future Directions

Further ideas to extend volatility normalization:

- Realized volatility from intraday data *(implemented)*
- GARCH-based volatility estimates for forecasting *(implemented)*
- Integration of implied volatility from options data *(implemented)*
- Cross-sectional normalization across multiple assets *(implemented)*
- Regime-switching models for volatility regimes *(implemented)*
- Event-driven features reacting to scheduled announcements
- Incremental computations for streaming data
- Visualization helpers for normalized indicators

