"""Smoke tests asserting the package is installed and importable."""

from __future__ import annotations

from importlib.resources import files

import local_rag


def test_version_is_exposed() -> None:
    assert local_rag.__version__ == "0.1.0"


def test_package_is_typed() -> None:
    """A ``py.typed`` marker must ship so downstream consumers get type information."""
    assert files("local_rag").joinpath("py.typed").is_file()
