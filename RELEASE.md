# Release Process

This project uses a tag-driven release flow.

## Preconditions

- `pyproject.toml` contains the version you intend to release.
- CI is green on the commit you plan to tag.
- PyPI trusted publishing is configured for this repository.

## Steps

1. Update `tool.poetry.version` in `pyproject.toml`.
2. Run the verification commands:

```bash
python -m black --check src tests
python -m ruff check .
python -m mypy .
pytest -q
```

3. Commit the version bump.
4. Create and push a tag in the form `vX.Y.Z`.

```bash
git tag v0.1.0
git push origin v0.1.0
```

5. The `Release` workflow verifies that the tag matches `pyproject.toml`, builds the source and wheel artifacts, uploads them to the workflow run, and publishes them to PyPI.

## Notes

- CI does not rewrite `pyproject.toml` or push commits.
- If the tag and `pyproject.toml` disagree, the release job fails.
- `workflow_dispatch` can be used to test the release workflow without publishing from a tag.
