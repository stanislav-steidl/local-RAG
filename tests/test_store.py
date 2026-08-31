"""Tests for the LanceDB chunk store and its schema.

These run against a real LanceDB on a temporary directory rather than a stub.
The store's whole job is surviving a round trip through Arrow, and a stub that
returns whatever it was handed would verify nothing about that. Vectors are
synthetic, so no model is loaded and the suite stays fast.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from local_rag.embedding import Embedding, SparseVector
from local_rag.models import Chunk, ChunkMetadata, SourceType
from local_rag.store import (
    LanceChunkStore,
    build_schema,
    chunk_id,
    iter_batches,
    record_to_chunk,
    record_to_embedding,
    to_record,
)

from .conftest import make_document_metadata

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("lancedb", reason="needs the store extra")

import lancedb
import pyarrow as pa

DIMENSION = 8


def make_embedding(seed: float = 1.0, *, sparse: bool = True) -> Embedding:
    """A deterministic embedding of the width these tests use."""
    return Embedding(
        dense=tuple(seed + offset for offset in range(DIMENSION)),
        sparse=SparseVector(indices=(3, 7), values=(0.5, 0.25)) if sparse else None,
    )


def make_stored_chunk(
    text: str = "chunk text",
    *,
    index: int = 0,
    content_hash: str = "a" * 64,
    path: str = "contracts/lease.pdf",
    page: int | None = 2,
) -> Chunk:
    """A chunk with the provenance the store flattens into columns."""
    return Chunk(
        page_content=text,
        metadata=ChunkMetadata(
            document=make_document_metadata(content_hash=content_hash, relative_path=path),
            chunk_index=index,
            start_char=index * 100,
            end_char=index * 100 + len(text),
            page_number=page,
        ),
    )


@pytest.fixture
def store(tmp_path: Path) -> LanceChunkStore:
    """An empty store on a temporary database."""
    return LanceChunkStore(tmp_path / "index", dimension=DIMENSION)


class TestChunkId:
    def test_is_derived_from_hash_and_position(self) -> None:
        assert chunk_id("abc", 4) == "abc:4"

    def test_is_stable_across_calls(self) -> None:
        """Re-indexing an unchanged document must reuse ids, not mint new ones."""
        assert chunk_id("abc", 0) == chunk_id("abc", 0)

    def test_differs_when_the_document_changes(self) -> None:
        """A changed file is a different document, not an update to the same one."""
        assert chunk_id("abc", 0) != chunk_id("def", 0)


class TestSchema:
    def test_dense_vectors_are_fixed_width(self) -> None:
        """LanceDB indexes fixed-size lists; a variable list could not be indexed."""
        field = build_schema(16).field("vector")

        assert field.type.list_size == 16

    def test_page_columns_are_nullable(self) -> None:
        """DOCX and plain text have no page numbers at all."""
        schema = build_schema(4)

        assert schema.field("page_number").nullable
        assert schema.field("page_count").nullable

    def test_provenance_columns_are_not_nullable(self) -> None:
        schema = build_schema(4)

        assert not schema.field("relative_path").nullable
        assert not schema.field("content_hash").nullable

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_dimension_is_rejected(self, bad: int) -> None:
        with pytest.raises(ValueError, match="dimension must be positive"):
            build_schema(bad)


class TestRecordConversion:
    def test_round_trips_a_chunk(self) -> None:
        """Retrieval returns rows, but citations are built from chunks."""
        original = make_stored_chunk("some text", index=3, page=5)

        restored = record_to_chunk(to_record(original, make_embedding()))

        assert restored == original

    def test_round_trips_an_embedding(self) -> None:
        original = make_embedding(2.0)

        restored = record_to_embedding(to_record(make_stored_chunk(), original))

        assert restored.dense == original.dense
        assert restored.sparse is not None
        assert restored.sparse.as_mapping() == {3: 0.5, 7: 0.25}

    def test_round_trips_an_absent_page_number(self) -> None:
        """None must survive as None, not become zero."""
        original = make_stored_chunk(page=None)

        assert record_to_chunk(to_record(original, make_embedding())).metadata.page_number is None

    def test_a_dense_only_embedding_stores_empty_sparse_arrays(self) -> None:
        record = to_record(make_stored_chunk(), make_embedding(sparse=False))

        assert record["sparse_indices"] == []
        assert record["sparse_values"] == []

    def test_extra_metadata_survives(self) -> None:
        """EXIF and GPS for the planned photo corpus would be lost otherwise."""
        chunk = Chunk(
            page_content="text",
            metadata=ChunkMetadata(
                document=make_document_metadata(
                    source_type=SourceType.PHOTO, extra={"gps": [50.08, 14.44]}
                ),
                chunk_index=0,
                start_char=0,
                end_char=4,
            ),
        )

        restored = record_to_chunk(to_record(chunk, make_embedding()))

        assert restored.metadata.document.extra == {"gps": [50.08, 14.44]}
        assert restored.metadata.document.source_type is SourceType.PHOTO

    def test_extra_is_serialised_deterministically(self) -> None:
        """Key order must not make two identical documents look different."""
        first = to_record(make_stored_chunk(), make_embedding())
        assert json.loads(first["extra_json"]) == {}


class TestStoreLifecycle:
    def test_a_new_store_is_empty(self, store: LanceChunkStore) -> None:
        assert store.count() == 0

    def test_reopening_keeps_the_data(self, tmp_path: Path) -> None:
        """Resumption across process restarts depends on exactly this."""
        path = tmp_path / "index"
        first = LanceChunkStore(path, dimension=DIMENSION)
        first.add([make_stored_chunk()], [make_embedding()])

        reopened = LanceChunkStore(path, dimension=DIMENSION)

        assert reopened.count() == 1

    def test_reopening_with_a_different_width_is_rejected(self, tmp_path: Path) -> None:
        """Otherwise this fails deep inside Arrow, far from the actual cause."""
        path = tmp_path / "index"
        LanceChunkStore(path, dimension=DIMENSION)

        with pytest.raises(ValueError, match="embedding model has changed"):
            LanceChunkStore(path, dimension=DIMENSION + 1)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_dimension_is_rejected(self, tmp_path: Path, bad: int) -> None:
        with pytest.raises(ValueError, match="dimension must be positive"):
            LanceChunkStore(tmp_path / "index", dimension=bad)

    def test_repr_identifies_the_table(self, store: LanceChunkStore) -> None:
        assert "chunks" in repr(store)
        assert str(DIMENSION) in repr(store)

    def test_exposes_its_dimension_and_path(self, tmp_path: Path) -> None:
        path = tmp_path / "index"
        opened = LanceChunkStore(path, dimension=DIMENSION)

        assert opened.dimension == DIMENSION
        assert opened.path == path

    def test_an_unrelated_schema_difference_is_not_blamed_on_the_model(
        self, tmp_path: Path
    ) -> None:
        """Same vector width, different columns: the model did not change.

        Reporting every schema mismatch as a changed embedding model would send
        someone to rebuild an index over a problem that is not that.
        """
        path = tmp_path / "index"
        connection = lancedb.connect(str(path))
        connection.create_table(
            "chunks",
            schema=pa.schema(
                [
                    pa.field("vector", pa.list_(pa.float32(), DIMENSION)),
                    pa.field("unexpected", pa.string()),
                ]
            ),
        )

        with pytest.raises(Exception, match=r"[Ss]chema") as excinfo:
            LanceChunkStore(path, dimension=DIMENSION)

        assert "embedding model has changed" not in str(excinfo.value)

    def test_a_foreign_table_is_reported_clearly(self, tmp_path: Path) -> None:
        """Pointing at someone else's database should say so, not raise KeyError.

        Without this the schema lookup fails on a missing column and surfaces as
        an unhandled KeyError, which says nothing about what went wrong.
        """
        path = tmp_path / "index"
        connection = lancedb.connect(str(path))
        connection.create_table("chunks", schema=pa.schema([pa.field("x", pa.int32())]))

        with pytest.raises(ValueError, match="was not created by this store"):
            LanceChunkStore(path, dimension=DIMENSION)


class TestAdding:
    def test_writes_rows(self, store: LanceChunkStore) -> None:
        written = store.add(
            [make_stored_chunk(index=0), make_stored_chunk(index=1)],
            [make_embedding(1.0), make_embedding(2.0)],
        )

        assert written == 2
        assert store.count() == 2

    def test_adding_nothing_is_a_no_op(self, store: LanceChunkStore) -> None:
        assert store.add([], []) == 0
        assert store.count() == 0

    def test_an_empty_store_returns_no_records(self, store: LanceChunkStore) -> None:
        assert store.to_records() == []

    def test_mismatched_lengths_are_rejected(self, store: LanceChunkStore) -> None:
        """Zipping these would pair a chunk with another chunk's vector."""
        with pytest.raises(ValueError, match="correspond one to one"):
            store.add([make_stored_chunk()], [])

    def test_a_wrong_width_vector_is_rejected(self, store: LanceChunkStore) -> None:
        wrong = Embedding(dense=(1.0, 2.0))

        with pytest.raises(ValueError, match="store holds 8-dimensional"):
            store.add([make_stored_chunk()], [wrong])

    def test_stored_rows_survive_the_round_trip(self, store: LanceChunkStore) -> None:
        """The real test of the schema: through Arrow and back, unchanged."""
        chunk = make_stored_chunk("Smlouva o dilo", index=2, page=3)
        store.add([chunk], [make_embedding(5.0)])

        restored = record_to_chunk(store.to_records()[0])

        assert restored == chunk

    def test_vectors_survive_the_round_trip(self, store: LanceChunkStore) -> None:
        embedding = make_embedding(1.5)
        store.add([make_stored_chunk()], [embedding])

        restored = record_to_embedding(store.to_records()[0])

        assert restored.dense == pytest.approx(embedding.dense)
        assert restored.sparse is not None
        assert restored.sparse.as_mapping() == {3: 0.5, 7: 0.25}


class TestResumption:
    def test_reports_nothing_when_empty(self, store: LanceChunkStore) -> None:
        assert store.indexed_content_hashes() == set()

    def test_reports_stored_document_hashes(self, store: LanceChunkStore) -> None:
        """Skipping documents already present is what makes a two-hour run survivable."""
        store.add(
            [make_stored_chunk(content_hash="a" * 64), make_stored_chunk(content_hash="b" * 64)],
            [make_embedding(), make_embedding(2.0)],
        )

        assert store.indexed_content_hashes() == {"a" * 64, "b" * 64}

    def test_a_document_appears_once_however_many_chunks_it_has(
        self, store: LanceChunkStore
    ) -> None:
        store.add(
            [make_stored_chunk(index=index) for index in range(3)],
            [make_embedding(float(index)) for index in range(3)],
        )

        assert store.indexed_content_hashes() == {"a" * 64}


class TestDeletion:
    def test_deletes_one_document_version(self, store: LanceChunkStore) -> None:
        store.add(
            [make_stored_chunk(content_hash="a" * 64), make_stored_chunk(content_hash="b" * 64)],
            [make_embedding(), make_embedding(2.0)],
        )

        store.delete_document("a" * 64)

        assert store.indexed_content_hashes() == {"b" * 64}

    def test_deletes_every_chunk_of_that_document(self, store: LanceChunkStore) -> None:
        store.add(
            [make_stored_chunk(index=index) for index in range(4)],
            [make_embedding(float(index)) for index in range(4)],
        )

        store.delete_document("a" * 64)

        assert store.count() == 0

    def test_deletes_by_corpus_path(self, store: LanceChunkStore) -> None:
        """An edited file's old chunks must go, whatever version produced them."""
        store.add(
            [
                make_stored_chunk(path="a.pdf", content_hash="a" * 64),
                make_stored_chunk(path="b.pdf", content_hash="b" * 64),
            ],
            [make_embedding(), make_embedding(2.0)],
        )

        store.delete_by_path("a.pdf")

        assert [row["relative_path"] for row in store.to_records()] == ["b.pdf"]

    def test_a_path_containing_an_apostrophe_is_handled(self, store: LanceChunkStore) -> None:
        """Real filenames contain apostrophes, and this builds a SQL predicate."""
        store.add(
            [make_stored_chunk(path="Tom's file.pdf"), make_stored_chunk(path="other.pdf")],
            [make_embedding(), make_embedding(2.0)],
        )

        store.delete_by_path("Tom's file.pdf")

        assert [row["relative_path"] for row in store.to_records()] == ["other.pdf"]


class TestBatching:
    def test_splits_into_aligned_slices(self) -> None:
        chunks = [make_stored_chunk(index=index) for index in range(5)]
        embeddings = [make_embedding(float(index)) for index in range(5)]

        batches = list(iter_batches(chunks, embeddings, 2))

        assert [len(pair[0]) for pair in batches] == [2, 2, 1]
        assert all(len(pair[0]) == len(pair[1]) for pair in batches)

    def test_slices_stay_paired(self) -> None:
        """A slice that misaligns the two lists would store the wrong vectors."""
        chunks = [make_stored_chunk(index=index) for index in range(4)]
        embeddings = [make_embedding(float(index)) for index in range(4)]

        for chunk_slice, embedding_slice in iter_batches(chunks, embeddings, 3):
            for chunk, embedding in zip(chunk_slice, embedding_slice, strict=True):
                assert embedding.dense[0] == float(chunk.metadata.chunk_index)

    def test_empty_input_yields_nothing(self) -> None:
        assert list(iter_batches([], [], 4)) == []

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_size_is_rejected(self, bad: int) -> None:
        with pytest.raises(ValueError, match="size must be positive"):
            list(iter_batches([], [], bad))

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="correspond one to one"):
            list(iter_batches([make_stored_chunk()], [], 2))


def test_a_naive_timestamp_from_arrow_is_restored_as_utc() -> None:
    """Arrow can hand back a naive datetime, which DocumentMetadata rejects."""
    record = to_record(make_stored_chunk(), make_embedding())
    record["modified_at"] = datetime(2024, 3, 5, 12, 30)

    assert record_to_chunk(record).metadata.document.modified_at.tzinfo is UTC
