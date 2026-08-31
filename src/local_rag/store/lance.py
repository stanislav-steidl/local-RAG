"""The LanceDB-backed chunk store.

Writing is the expensive half of this pipeline: embedding a corpus of a few
thousand chunks costs hours on a CPU. The store is therefore built around
resumption rather than around throughput — it can say which documents it
already holds, so a run that dies at ninety percent picks up where it stopped.

That guarantee is only worth having if it is exact, which drives two decisions.
Writes are **per document and atomic**: a document's chunks are committed in a
single operation, so a hash that is present is a document that is complete.
And writes are **upserts** keyed on the derived chunk id, so retrying a
document replaces its rows instead of duplicating them.

Search lives in the retrieval stage, not here. This module answers "what is
stored, and how do I change it"; ranking is a separate concern.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from local_rag.errors import optional_dependency
from local_rag.store.schema import build_schema, to_record

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from local_rag.embedding import Embedding
    from local_rag.models import Chunk

__all__ = ["DEFAULT_TABLE_NAME", "LanceChunkStore"]

logger = logging.getLogger(__name__)

#: Name of the table chunks live in. A constant rather than a parameter because
#: nothing yet needs two tables in one database, and a name that can vary is a
#: name that can be got wrong.
DEFAULT_TABLE_NAME = "chunks"

#: LanceDB signals an absent table with a plain ValueError, so the message is
#: the only thing distinguishing it from a permission or I/O failure.
_TABLE_NOT_FOUND = "not found"


class LanceChunkStore:
    """Stores embedded chunks in an embedded LanceDB database.

    The database is a directory on disk. Opening one that does not exist
    creates it; opening one that does reuses it, which is what makes indexing
    resumable across process restarts.
    """

    def __init__(
        self,
        path: Path,
        *,
        dimension: int,
        table_name: str = DEFAULT_TABLE_NAME,
    ) -> None:
        """Open or create a store.

        Args:
            path: Directory holding the database.
            dimension: Width of the dense vectors to be stored. A table built
                for one width cannot hold another's vectors, so this is fixed
                at creation and checked on reopen.
            table_name: Table to read and write.

        Raises:
            ValueError: If ``dimension`` is not positive, or an existing table
                was built for a different width.
            MissingDependencyError: If LanceDB is not installed.
        """
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")

        with optional_dependency(
            "lancedb",
            "the vector store requires LanceDB: pip install 'local-rag[store]'",
        ):
            import lancedb  # noqa: PLC0415  # optional dependency, imported on use

        self._path = path
        self._dimension = dimension
        self._table_name = table_name
        self._connection = lancedb.connect(str(path))

        # Open first, create only on a genuine absence. Creating with
        # `exist_ok` instead would surface a width mismatch as LanceDB's own
        # "schemas disagree", and catching that broadly would swallow
        # permission and I/O failures too.
        try:
            self._table = self._connection.open_table(table_name)
        except ValueError as error:
            if _TABLE_NOT_FOUND not in str(error).lower():
                raise
            self._table = self._connection.create_table(table_name, schema=build_schema(dimension))
            logger.debug("Created table %r in %s", table_name, path)
        else:
            self._check_schema()

    def _check_schema(self) -> None:
        """Reject an existing table this store cannot write to.

        Opening rather than creating means LanceDB never compares schemas for
        us, so a mismatch would otherwise surface as an Arrow error on the
        first write — long after the point where it could be explained. The
        vector width is checked first because a changed embedding model is by
        far the likeliest cause and has a specific remedy.

        Raises:
            ValueError: If the table stores vectors of a different width, or
                does not have the columns this store writes.
        """
        stored = self._table.schema
        expected = build_schema(self._dimension)

        if "vector" in stored.names:
            width = getattr(stored.field("vector").type, "list_size", None)
            if width is not None and width != self._dimension:
                raise ValueError(
                    f"table {self._table_name!r} stores {width}-dimensional vectors but "
                    f"{self._dimension} was requested; the embedding model has changed "
                    f"and the index must be rebuilt"
                )

        missing = [field.name for field in expected if field.name not in stored.names]
        if missing:
            raise ValueError(
                f"table {self._table_name!r} in {self._path} is missing columns "
                f"{missing}, so it was not created by this store; point at a different "
                f"directory or table"
            )

    @property
    def dimension(self) -> int:
        """Width of the dense vectors this store holds."""
        return self._dimension

    @property
    def path(self) -> Path:
        """Directory the database lives in."""
        return self._path

    def count(self) -> int:
        """Number of chunks stored."""
        return int(self._table.count_rows())

    def add_document(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> int:
        """Store every chunk of one document, in a single commit.

        One document per call, deliberately. Splitting a document across
        several writes would let a crash leave it half stored, and since
        :meth:`indexed_content_hashes` reports a document as done once its rows
        exist, the remainder would be skipped for good on the next run —
        silently losing content rather than failing.

        Rows are upserted on ``chunk_id``, so retrying a document that was
        partially or fully written replaces its rows rather than duplicating
        them. LanceDB does not enforce uniqueness itself; a derived identifier
        alone would not prevent duplicates.

        Args:
            chunks: Every chunk of one document.
            embeddings: Their embeddings, in the same order.

        Returns:
            The number of rows written.

        Raises:
            ValueError: If the sequences differ in length, an embedding is not
                the width this store holds, or the chunks come from more than
                one document.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"got {len(chunks)} chunks and {len(embeddings)} embeddings; "
                f"they must correspond one to one"
            )
        if not chunks:
            return 0

        hashes = {chunk.metadata.document.content_hash for chunk in chunks}
        if len(hashes) > 1:
            raise ValueError(
                f"add_document stores one document per call, but was given {len(hashes)}; "
                f"writing them together would make partial writes indistinguishable "
                f"from complete ones"
            )

        wrong = sorted(
            {
                embedding.dimension
                for embedding in embeddings
                if embedding.dimension != self._dimension
            }
        )
        if wrong:
            raise ValueError(
                f"store holds {self._dimension}-dimensional vectors but was given {wrong}"
            )

        rows = [
            to_record(chunk, embedding) for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        (
            self._table.merge_insert("chunk_id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )
        return len(rows)

    def indexed_content_hashes(self) -> set[str]:
        """Return the content hashes of documents already stored.

        This is what makes an interrupted run resumable: the indexer skips any
        document whose hash is present, and a document that changed hashes
        differently, so it is never mistaken for one that is done. The answer
        is only trustworthy because :meth:`add_document` commits a whole
        document at once.

        Returns:
            Every distinct ``content_hash`` in the table.
        """
        if self.count() == 0:
            return set()

        # Project a single column. `to_arrow()` would materialise every dense
        # and sparse vector merely to read a string, which on the multi-gigabyte
        # indexes this design targets means reading the whole index to answer a
        # question about its keys. `limit(0)` means no limit; the default is 10,
        # which would silently truncate.
        projected = self._table.search(None).select(["content_hash"]).limit(0).to_arrow()
        return {str(value) for value in projected.column("content_hash").to_pylist()}

    def delete_document(self, content_hash: str) -> None:
        """Remove every chunk belonging to one version of a document.

        Keyed by hash rather than path so that re-indexing a modified file
        leaves its previous chunks behind for explicit removal, rather than
        silently mixing two versions of the same document in the index.

        Args:
            content_hash: Digest of the document whose chunks to delete.
        """
        self._table.delete(f"content_hash = '{_escape(content_hash)}'")

    def delete_by_path(self, relative_path: str) -> None:
        """Remove every chunk that came from one corpus path.

        Used when a file is edited or deleted: its old chunks must go, whatever
        version they were built from.

        Args:
            relative_path: Corpus-relative path whose chunks to delete.
        """
        self._table.delete(f"relative_path = '{_escape(relative_path)}'")

    def to_records(self) -> list[dict[str, Any]]:
        """Return every stored row.

        Intended for tests and small corpora; retrieval reads through a query
        rather than materialising the table.
        """
        if self.count() == 0:
            return []
        return list(self._table.to_arrow().to_pylist())

    def __repr__(self) -> str:
        """Identify the table and where it lives."""
        return (
            f"{type(self).__name__}(path={str(self._path)!r}, "
            f"table={self._table_name!r}, dimension={self._dimension})"
        )


def _escape(value: str) -> str:
    """Escape a value for a LanceDB SQL predicate.

    Args:
        value: Literal to embed in a predicate.

    Returns:
        The value with single quotes doubled, so a path containing an
        apostrophe cannot terminate the string early.
    """
    return value.replace("'", "''")
