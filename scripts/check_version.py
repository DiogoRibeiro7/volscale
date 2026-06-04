"""Verify that the project version matches a git tag.

This script verifies that the version declared in ``pyproject.toml`` matches the
provided git tag.
"""

from __future__ import annotations

import argparse


def get_pyproject_version() -> str:
    """Return the version declared in ``pyproject.toml``."""

    with open("pyproject.toml", "rb") as pyproject:
        try:
            import tomllib

            data = tomllib.load(pyproject)
        except ModuleNotFoundError:  # pragma: no cover - fallback for older Python
            import tomli

            pyproject.seek(0)
            data = tomli.load(pyproject)
    return data["tool"]["poetry"]["version"]


def main(tag: str) -> None:
    """Ensure the project version matches the git tag.

    Args:
        tag: Git tag string, optionally prefixed with ``v``.
    """

    version = get_pyproject_version()
    tag_version = tag.lstrip("v")
    if version != tag_version:
        raise SystemExit(f"Version mismatch: pyproject={version} tag={tag_version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensure that pyproject version matches the provided tag."
    )
    parser.add_argument("tag", help="Git tag string, e.g. v1.2.3")
    args = parser.parse_args()
    main(args.tag)
