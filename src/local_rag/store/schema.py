"""The columnar schema chunks are stored in, and conversion to and from it.

LanceDB stores Arrow tables, so the nested Python objects the pipeline passes
around have to be flattened into columns. Keeping that translation in one place
means the store itself deals only in rows, and the flattening is testable
without a database.

Two choices shape the schema:

*Deterministic row identity.* A chunk's id is derived from its document's
content hash and its position, not generated. Re-indexing an unchanged file
therefore produces the same ids, which is what lets an interrupted run resume
rather than duplicate — and at roughly six seconds per chunk on CPU, resuming
is not a refinement.

*Metadata as columns, not a blob.* Arrow's columnar layout is the reason
LanceDB was chosen over an index that only stores vectors: filtering by date or
path stays cheap at multi-gigabyte scale. Only the open-ended ``extra`` mapping
is serialised, since its keys are not known ahead of time.
"""

from __future__ import annotations

import json
from datetime import UTC
from typing import TYPE_CHECKING, Any

from local_rag.embedding import Embedding, SparseVector
from local_rag.models import Chunk, ChunkMetadata, DocumentMetadata, SourceType

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = [
    "EMBEDDING_DIMENSION_KEY",
    "EMBEDDING_MODEL_KEY",
    "build_schema",
    "chunk_id",
    "record_to_chunk",
    "record_to_embedding",
    "to_record",
]

#: Arrow schema metadata keys recording which embedder built a table. Bytes,
#: because that is what Arrow metadata holds.
EMBEDDING_MODEL_KEY = b"local_rag.embedding_model"
EMBEDDING_DIMENSION_KEY = b"local_rag.embedding_dimension"


def chunk_id(content_hash: str, chunk_index: int) -> str:
    """Return the stable identity of one chunk.

    Derived rather than generated, so re-indexing an unchanged document yields
    the same ids and can replace its rows instead of duplicating them. The
    content hash rather than the path, because a file that moves is the same
    document while a file that changes is not.

    Args:
        content_hash: Digest of the source document's bytes.
        chunk_index: Position of the chunk within that document.

    Returns:
        An identifier unique to this chunk of this version of this document.
    """
    return f"{content_hash}:{chunk_index}"


def build_schema(dimension: int, model_id: str) -> pa.Schema:
    """Describe the table layout for a given embedder.

    Both the width and the model identity are recorded. Width alone does not
    identify an embedding model: two different models can produce 1024-wide
    vectors that mean entirely different things, and mixing them in one table
    yields a store whose similarity scores compare incomparable spaces while
    every structural check passes. The identity is carried in the schema's
    Arrow metadata, which LanceDB preserves across reopen.

    Args:
        dimension: Length of the dense vectors to be stored.
        model_id: Identifier of the embedder that produces them.

    Returns:
        The Arrow schema for the chunk table.

    Raises:
        ValueError: If ``dimension`` is not positive or ``model_id`` is empty.
    """
    import pyarrow as pa  # noqa: PLC0415  # optional dependency, imported on use

    if dimension <= 0:
        raise ValueError(f"dimension must be positive, got {dimension}")
    if not model_id:
        raise ValueError("model_id must not be empty")

    return pa.schema(
        [
            pa.field("chunk_id", pa.string(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            # Fixed width: LanceDB builds its vector index from this, and a
            # variable-length list could not be indexed.
            pa.field("vector", pa.list_(pa.float32(), dimension), nullable=False),
            # Sparse vectors are stored as parallel arrays rather than a map,
            # matching how SparseVector already holds them and how a lexical
            # scorer wants to read them.
            pa.field("sparse_indices", pa.list_(pa.int32()), nullable=False),
            pa.field("sparse_values", pa.list_(pa.float32()), nullable=False),
            # Document provenance, one column each so filters stay cheap.
            pa.field("relative_path", pa.string(), nullable=False),
            pa.field("source_type", pa.string(), nullable=False),
            pa.field("file_extension", pa.string(), nullable=False),
            pa.field("size_bytes", pa.int64(), nullable=False),
            pa.field("modified_at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("content_hash", pa.string(), nullable=False),
            pa.field("page_count", pa.int32(), nullable=True),
            # Position within the document, enough to locate the chunk again.
            pa.field("chunk_index", pa.int32(), nullable=False),
            pa.field("start_char", pa.int32(), nullable=False),
            pa.field("end_char", pa.int32(), nullable=False),
            pa.field("page_number", pa.int32(), nullable=True),
            # Open-ended metadata — EXIF and GPS for the planned photo corpus —
            # whose keys are not known in advance, so it cannot be columns.
            pa.field("extra_json", pa.string(), nullable=False),
        ],
        metadata={
            EMBEDDING_MODEL_KEY: model_id.encode(),
            EMBEDDING_DIMENSION_KEY: str(dimension).encode(),
        },
    )


def to_record(chunk: Chunk, embedding: Embedding) -> dict[str, Any]:
    """Flatten a chunk and its embedding into one table row.

    Args:
        chunk: The chunk to store.
        embedding: Its vectors. A dense-only embedding is stored with empty
            sparse arrays, which read back as an empty :class:`SparseVector`.

    Returns:
        A row matching :func:`build_schema`.

    Raises:
        ValueError: If the document's ``extra`` mapping holds values JSON
            cannot represent. Coercing them instead would let provenance change
            type silently between what was stored and what comes back.

    Note:
        ``extra`` survives a round trip only for JSON-native values. A tuple is
        stored and returned as a list, since JSON has one sequence type; the
        mapping is otherwise unchanged.
    """
    document = chunk.metadata.document
    sparse = embedding.sparse or SparseVector()

    return {
        "chunk_id": chunk_id(document.content_hash, chunk.metadata.chunk_index),
        "text": chunk.page_content,
        "vector": list(embedding.dense),
        "sparse_indices": list(sparse.indices),
        "sparse_values": list(sparse.values),
        "relative_path": document.relative_path,
        "source_type": document.source_type.value,
        "file_extension": document.file_extension,
        "size_bytes": document.size_bytes,
        "modified_at": document.modified_at,
        "content_hash": document.content_hash,
        "page_count": document.page_count,
        "chunk_index": chunk.metadata.chunk_index,
        "start_char": chunk.metadata.start_char,
        "end_char": chunk.metadata.end_char,
        "page_number": chunk.metadata.page_number,
        "extra_json": _dump_extra(document.extra, document.relative_path),
    }


def _dump_extra(extra: dict[str, Any], relative_path: str) -> str:
    """Serialise a document's open-ended metadata, refusing to guess.

    ``default=str`` would make anything serialisable, at the cost of turning a
    datetime or a Decimal into a string that reads back as a string — the type
    changing between store and retrieve, with nothing to indicate it happened.
    An error at write time is recoverable; provenance that quietly changed type
    is not detectable at all.

    Args:
        extra: The mapping to serialise.
        relative_path: Document the mapping belongs to, for the error message.

    Returns:
        Its JSON representation, with keys sorted so equal mappings serialise
        identically.

    Raises:
        ValueError: If any value is not JSON-native.
    """
    try:
        return json.dumps(extra, sort_keys=True)
    except TypeError as error:
        raise ValueError(
            f"metadata 'extra' for {relative_path} holds a value JSON cannot represent "
            f"({error}); store JSON-native values so provenance survives unchanged"
        ) from error


def record_to_chunk(record: dict[str, Any]) -> Chunk:
    """Rebuild a chunk from a stored row.

    Retrieval returns rows, but the rest of the pipeline — citations above all —
    speaks in chunks, so the flattening has to be reversible.

    Args:
        record: A row as returned by the store.

    Returns:
        The chunk it was built from.
    """
    modified_at = record["modified_at"]
    if modified_at.tzinfo is None:
        # Arrow may hand back a naive datetime even for a tz-aware column, and
        # DocumentMetadata rejects those outright.
        modified_at = modified_at.replace(tzinfo=UTC)

    document = DocumentMetadata(
        relative_path=record["relative_path"],
        source_type=SourceType(record["source_type"]),
        file_extension=record["file_extension"],
        size_bytes=int(record["size_bytes"]),
        modified_at=modified_at,
        content_hash=record["content_hash"],
        page_count=None if record["page_count"] is None else int(record["page_count"]),
        extra=json.loads(record["extra_json"]),
    )

    return Chunk(
        page_content=record["text"],
        metadata=ChunkMetadata(
            document=document,
            chunk_index=int(record["chunk_index"]),
            start_char=int(record["start_char"]),
            end_char=int(record["end_char"]),
            page_number=None if record["page_number"] is None else int(record["page_number"]),
        ),
    )


def record_to_embedding(record: dict[str, Any]) -> Embedding:
    """Rebuild an embedding from a stored row.

    Args:
        record: A row as returned by the store.

    Returns:
        The embedding it holds. Empty sparse arrays become an empty
        :class:`SparseVector` rather than ``None``, since a row written by a
        dense-only backend is indistinguishable from one whose terms were all
        dropped.
    """
    return Embedding(
        dense=tuple(float(value) for value in record["vector"]),
        sparse=SparseVector(
            indices=tuple(int(index) for index in record["sparse_indices"]),
            values=tuple(float(value) for value in record["sparse_values"]),
        ),
    )
