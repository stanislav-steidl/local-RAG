"""Persistence for embedded chunks.

The store keeps chunks, their vectors and their provenance in one on-disk
LanceDB table. Importing this package does not require LanceDB: it is behind
the ``store`` extra and loaded when a store is opened.
"""

from __future__ import annotations

from local_rag.store.lance import DEFAULT_TABLE_NAME, LanceChunkStore
from local_rag.store.schema import (
    build_schema,
    chunk_id,
    record_to_chunk,
    record_to_embedding,
    to_record,
)

__all__ = [
    "DEFAULT_TABLE_NAME",
    "LanceChunkStore",
    "build_schema",
    "chunk_id",
    "record_to_chunk",
    "record_to_embedding",
    "to_record",
]
