# Contributing

## Scope

The repository separates stable utilities from exploratory helpers.

- Stable APIs live in the top-level `volnorm` package.
- Experimental APIs live under `volnorm.experimental`.
- Compatibility wrappers exist for some older import paths, but new work should target the explicit stable or experimental boundary directly.

## Development Setup

Install dependencies with Poetry:

```bash
poetry install
```

## Verification

Run the same checks used in CI before opening a pull request:

```bash
python -m black --check src tests
python -m ruff check .
python -m mypy .
pytest -q
```

If you change packaging or release logic, also build artifacts locally:

```bash
poetry build
python scripts/check_version.py v0.1.1
```

Replace `v0.1.1` with the version you are validating.

## Coding Guidelines

- Keep stable and experimental code clearly separated.
- Add or update tests for behavior changes.
- Prefer file-backed reference fixtures for numerical benchmarks.
- Treat the CLI as a user-facing product surface, not just a thin script.
- Do not make CI mutate repository state during verification or release.

## Pull Requests

Good pull requests should include:

- A clear description of the behavior change.
- Updated tests.
- Updated docs when the public API, CLI, or release flow changes.

## Releases

See [RELEASE.md](RELEASE.md) for the tag-driven release process.
