# Changelog

All notable changes to this project will be documented in this file.

The format is intentionally simple and keeps changes grouped by release.

## [0.1.1] - 2026-06-06

### Added

- Stable core package structure for volatility-scaled feature engineering.
- Dedicated `volscale.experimental` package for exploratory models and helpers.
- Typed `FeatureConfig` for feature-pipeline configuration.
- Release workflow with wheel/sdist build verification.
- File-backed numerical benchmark fixtures for ATR, implied volatility, and realized volatility.
- More capable CLI with JSON output, separator controls, index preservation, sorting, and row filtering.

### Changed

- Tightened validation across core APIs and CLI error handling.
- Narrowed the stable public surface and moved weaker routines behind experimental boundaries.
- Split CI verification from release publishing.

### Deprecated

- Legacy wrapper imports through `volscale.events`, `volscale.smoothing`, and `volscale.synthetic`.

## Unreleased

- No unreleased changes yet.
