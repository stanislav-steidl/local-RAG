"""local-RAG: a fully offline retrieval-augmented generation system.

The package is organised as a pipeline of small, independently testable stages:
ingestion, chunking, embedding, storage, retrieval and generation. Each stage
depends only on the plain data structures defined in :mod:`local_rag.models`,
so any stage can be swapped without disturbing its neighbours.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
