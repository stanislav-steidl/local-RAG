"""Turning text into the vectors the index searches over.

Importing this package pulls in no machine-learning dependency. Backends live
in their own modules and load their libraries on first use, so the rest of the
pipeline can depend on :class:`Embedder` while tests substitute a deterministic
fake and CI stays free of a multi-gigabyte download.
"""

from __future__ import annotations

from local_rag.embedding.base import (
    DEFAULT_BATCH_SIZE,
    Embedder,
    Embedding,
    SparseVector,
)

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "Embedder",
    "Embedding",
    "SparseVector",
]
