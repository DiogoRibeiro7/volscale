"""Synchronize the project version with a git tag.

This script verifies that the version declared in ``pyproject.toml`` matches the
provided git tag. When the ``--update`` flag is supplied, the version in
``pyproject.toml`` is updated to match the tag using Poetry.
"""

from __future__ import annotations

import argparse
import subprocess

import tomli


def get_pyproject_version() -> str:
    """Return the version declared in ``pyproject.toml``."""

    with open("pyproject.toml", "rb") as pyproject:
        data = tomli.load(pyproject)
    return data["tool"]["poetry"]["version"]


def set_pyproject_version(new_version: str) -> None:
    """Update ``pyproject.toml`` using Poetry.

    Args:
        new_version: The version string to write into the project file.
    """

    subprocess.run(["poetry", "version", new_version], check=True)


def main(tag: str, update: bool) -> None:
    """Ensure the project version matches the git tag.

    Args:
        tag: Git tag string, optionally prefixed with ``v``.
        update: Whether to update ``pyproject.toml`` on mismatch.
    """

    version = get_pyproject_version()
    tag_version = tag.lstrip("v")
    if version != tag_version:
        if update:
            set_pyproject_version(tag_version)
        else:
            raise SystemExit(f"Version mismatch: pyproject={version} tag={tag_version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensure that pyproject version matches the provided tag."
    )
    parser.add_argument("tag", help="Git tag string, e.g. v1.2.3")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update pyproject.toml when versions do not match",
    )
    args = parser.parse_args()
    main(args.tag, args.update)
