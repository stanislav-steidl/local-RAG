"""The LanceDB-backed chunk store.

Writing is the expensive half of this pipeline: embedding a corpus of a few
thousand chunks costs hours on a CPU. The store is therefore built around
resumption rather than around throughput — it can say what it already holds, so
a run that dies at ninety percent picks up where it stopped instead of starting
over.

Search lives in the retrieval stage, not here. This module answers "what is
stored, and how do I change it"; ranking is a separate concern with its own
design questions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from local_rag.errors import optional_dependency
from local_rag.store.schema import build_schema, to_record

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from local_rag.embedding import Embedding
    from local_rag.models import Chunk

__all__ = ["DEFAULT_TABLE_NAME", "LanceChunkStore"]

logger = logging.getLogger(__name__)

#: Name of the table chunks live in. A constant rather than a parameter because
#: nothing yet needs two tables in one database, and a name that can vary is a
#: name that can be got wrong.
DEFAULT_TABLE_NAME = "chunks"


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

        # `exist_ok` opens the table when it is already there and creates it
        # otherwise, which is exactly the open-or-create this needs. The
        # alternative — listing tables first — reads a paginated response, so a
        # database with many tables could report a table absent merely because
        # it fell beyond the first page.
        try:
            self._table = self._connection.create_table(
                table_name, schema=build_schema(dimension), exist_ok=True
            )
        except Exception:
            # An existing table whose schema differs. LanceDB reports only that
            # the schemas disagree, which does not say what to do about it; the
            # overwhelmingly likely cause is a changed embedding model, so open
            # the table and say so in those terms.
            self._table = self._connection.open_table(table_name)
            self._check_dimension()
            raise  # Same width, so the schemas differ for some other reason.

        self._check_dimension()
        logger.debug("Opened table %r in %s", table_name, path)

    def _check_dimension(self) -> None:
        """Reject a table whose vectors are a different width than expected.

        Writing 1024-wide vectors into a table built for 768 fails somewhere
        inside Arrow with a message about list lengths. Catching it here says
        what actually happened: the embedding model changed, and the index
        needs rebuilding.

        Raises:
            ValueError: If the stored width differs from the configured one, or
                the table has no vector column at all.
        """
        try:
            field = self._table.schema.field("vector")
        except KeyError:
            raise ValueError(
                f"table {self._table_name!r} in {self._path} has no 'vector' column, so it "
                f"was not created by this store; point at a different directory or table"
            ) from None

        stored = getattr(field.type, "list_size", None)
        if stored is not None and stored != self._dimension:
            raise ValueError(
                f"table {self._table_name!r} stores {stored}-dimensional vectors but "
                f"{self._dimension} was requested; the embedding model has changed and "
                f"the index must be rebuilt"
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

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> int:
        """Store chunks together with their embeddings.

        Args:
            chunks: Chunks to store.
            embeddings: Their embeddings, in the same order.

        Returns:
            The number of rows written.

        Raises:
            ValueError: If the two sequences differ in length, or an embedding
                is not the width this store was built for. Both would otherwise
                surface as an Arrow error far from the cause — or worse, pair a
                chunk with another chunk's vector.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"got {len(chunks)} chunks and {len(embeddings)} embeddings; "
                f"they must correspond one to one"
            )
        if not chunks:
            return 0

        wrong = [
            embedding.dimension
            for embedding in embeddings
            if embedding.dimension != self._dimension
        ]
        if wrong:
            raise ValueError(
                f"store holds {self._dimension}-dimensional vectors but was given "
                f"{sorted(set(wrong))}"
            )

        rows = [
            to_record(chunk, embedding) for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self._table.add(rows)
        return len(rows)

    def indexed_content_hashes(self) -> set[str]:
        """Return the content hashes already stored.

        This is what makes an interrupted run resumable: the indexer skips any
        document whose hash is already present, and a document that changed has
        a different hash, so it is not mistaken for one that is done.

        Returns:
            Every distinct ``content_hash`` in the table.
        """
        if self.count() == 0:
            return set()
        table = self._table.to_arrow()
        return {str(value) for value in table.column("content_hash").to_pylist()}

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


def iter_batches(
    chunks: Sequence[Chunk], embeddings: Sequence[Embedding], size: int
) -> Iterable[tuple[Sequence[Chunk], Sequence[Embedding]]]:
    """Yield aligned slices of chunks and embeddings.

    Writing an entire corpus in one call would hold every vector in memory at
    once; a few thousand 1024-wide float32 vectors is tolerable, a few hundred
    thousand is not.

    Args:
        chunks: Chunks to slice.
        embeddings: Their embeddings, in the same order.
        size: Maximum rows per slice.

    Yields:
        Pairs of corresponding slices.

    Raises:
        ValueError: If ``size`` is not positive or the inputs differ in length.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"got {len(chunks)} chunks and {len(embeddings)} embeddings; "
            f"they must correspond one to one"
        )

    for offset in range(0, len(chunks), size):
        yield chunks[offset : offset + size], embeddings[offset : offset + size]
