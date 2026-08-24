"""Errors shared across the pipeline.

Most failures belong to one stage and are defined alongside it. The two here
are not: every caller benefits from being able to catch anything this package
raises, and more than one stage defers a heavy optional dependency to first use
and has to report its absence the same way.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["LocalRagError", "MissingDependencyError", "optional_dependency"]


class LocalRagError(Exception):
    """Base class for every error raised by this package."""


class MissingDependencyError(LocalRagError):
    """An optional dependency required for the requested operation is absent.

    Heavy libraries — PDF parsers, PyTorch — live behind extras and are
    imported on first use, so their absence surfaces when something is actually
    attempted rather than at import time. The message names the extra that
    would fix it.
    """


@contextmanager
def optional_dependency(
    module: str,
    hint: str,
    *,
    error_type: type[MissingDependencyError] = MissingDependencyError,
) -> Iterator[None]:
    """Translate the absence of ``module`` into an actionable error.

    Wrap a deferred import. Only the named module going missing is treated as
    an absent extra: an installed package that fails its *own* imports raises
    ``ModuleNotFoundError`` too, naming the transitive module it could not
    find, and reporting that as "not installed" would send someone to reinstall
    something already present while hiding what actually broke.

    A submodule counts as the module itself — if ``FlagEmbedding.inference``
    cannot be found, the extra is unusable either way.

    Args:
        module: Top-level module the extra provides.
        hint: Message naming the extra to install.
        error_type: Exception to raise. Stages with their own hierarchy pass a
            subclass so that handlers catching that hierarchy keep working.

    Yields:
        Nothing; the block performs the import.

    Raises:
        MissingDependencyError: Or the given subclass, if ``module`` is absent.
    """
    try:
        yield
    except ModuleNotFoundError as error:
        if (error.name or "").partition(".")[0] != module:
            raise
        raise error_type(hint) from error
