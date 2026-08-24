"""Errors shared across the pipeline.

Most failures belong to one stage and are defined alongside it. The two here
are not: every caller benefits from being able to catch anything this package
raises, and more than one stage defers a heavy optional dependency to first use
and has to report its absence the same way.
"""

from __future__ import annotations

__all__ = ["LocalRagError", "MissingDependencyError"]


class LocalRagError(Exception):
    """Base class for every error raised by this package."""


class MissingDependencyError(LocalRagError):
    """An optional dependency required for the requested operation is absent.

    Heavy libraries — PDF parsers, PyTorch — live behind extras and are
    imported on first use, so their absence surfaces when something is actually
    attempted rather than at import time. The message names the extra that
    would fix it.
    """
