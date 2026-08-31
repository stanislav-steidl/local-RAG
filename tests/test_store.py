"""Tests for the LanceDB chunk store and its schema.

These run against a real LanceDB on a temporary directory rather than a stub.
The store's whole job is surviving a round trip through Arrow, and a stub that
returns whatever it was handed would verify nothing about that. Vectors are
synthetic, so no model is loaded and the suite stays fast.
"""

from __future__ import annotations

import json
import zoneinfo
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from local_rag.embedding import Embedding, SparseVector
from local_rag.models import Chunk, ChunkMetadata, SourceType
from local_rag.store import (
    IndexProvenance,
    LanceChunkStore,
    build_schema,
    chunk_id,
    record_to_chunk,
    record_to_embedding,
    to_record,
)

from .conftest import make_document_metadata

if TYPE_CHECKING:
    from pathlib import Path

lancedb = pytest.importorskip("lancedb", reason="needs the store extra")
pa = pytest.importorskip("pyarrow", reason="needs the store extra")

DIMENSION = 8
MODEL_ID = "test/embedder@max_length=512"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def provenance(**overrides: object) -> IndexProvenance:
    """The settings these tests build indexes with."""
    defaults: dict[str, object] = {
        "embedder_fingerprint": MODEL_ID,
        "dimension": DIMENSION,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    return IndexProvenance(**{**defaults, **overrides})  # type: ignore[arg-type]


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
    return LanceChunkStore(tmp_path / "index", provenance())


def test_the_timezone_database_is_available() -> None:
    """Arrow resolves the schema's UTC timestamps through zoneinfo.

    Windows ships no IANA database, so without the tzdata dependency every read
    of the table raises ArrowInvalid — which passed locally, where tzdata had
    arrived transitively, and failed only on a clean Windows runner.
    """
    assert zoneinfo.ZoneInfo("UTC") is not None


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
        field = build_schema(provenance(dimension=16)).field("vector")

        assert field.type.list_size == 16

    def test_page_columns_are_nullable(self) -> None:
        """DOCX and plain text have no page numbers at all."""
        schema = build_schema(provenance(dimension=4))

        assert schema.field("page_number").nullable
        assert schema.field("page_count").nullable

    def test_provenance_columns_are_not_nullable(self) -> None:
        schema = build_schema(provenance(dimension=4))

        assert not schema.field("relative_path").nullable
        assert not schema.field("content_hash").nullable

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_dimension_is_rejected(self, bad: int) -> None:
        with pytest.raises(ValueError, match="dimension must be positive"):
            build_schema(provenance(dimension=bad))

    def test_an_empty_model_id_is_rejected(self) -> None:
        """An unnamed embedder cannot be checked against on reopen."""
        with pytest.raises(ValueError, match="embedder_fingerprint must not be empty"):
            build_schema(provenance(embedder_fingerprint=""))

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_chunk_size_is_rejected(self, bad: int) -> None:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            build_schema(provenance(dimension=4, chunk_size=bad, chunk_overlap=0))

    @pytest.mark.parametrize("bad", [-1, 100, 200])
    def test_an_unusable_overlap_is_rejected(self, bad: int) -> None:
        """Recording chunking that could not have produced the rows is worse than none."""
        with pytest.raises(ValueError, match="chunk_overlap"):
            build_schema(provenance(dimension=4, chunk_size=100, chunk_overlap=bad))

    def test_the_chunking_settings_are_recorded(self) -> None:
        metadata = build_schema(
            provenance(dimension=4, chunk_size=1200, chunk_overlap=200)
        ).metadata

        assert metadata[b"local_rag.chunk_size"] == b"1200"
        assert metadata[b"local_rag.chunk_overlap"] == b"200"

    def test_the_embedder_is_recorded_in_the_schema(self) -> None:
        """Reopening compares against this; width alone does not identify a model."""
        metadata = build_schema(
            provenance(dimension=4, embedder_fingerprint="BAAI/bge-m3")
        ).metadata

        assert metadata[b"local_rag.embedder_fingerprint"] == b"BAAI/bge-m3"
        assert metadata[b"local_rag.embedding_dimension"] == b"4"


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

    def test_extra_metadata_survives_the_database(self, store: LanceChunkStore) -> None:
        """Through Arrow and LanceDB, not merely the two conversion helpers.

        Composing to_record with record_to_chunk in memory would pass even if
        the column were dropped from the schema, which is exactly the
        regression this is meant to catch.
        """
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

        store.add_document([chunk], [make_embedding()])
        restored = record_to_chunk(store.to_records()[0])

        assert restored.metadata.document.extra == {"gps": [50.08, 14.44]}
        assert restored.metadata.document.source_type is SourceType.PHOTO

    def test_extra_is_serialised_deterministically(self) -> None:
        """Key order must not make two identical documents look different."""
        first = to_record(make_stored_chunk(), make_embedding())
        assert json.loads(first["extra_json"]) == {}

    def test_metadata_json_cannot_represent_is_rejected(self) -> None:
        """Coercing it would change provenance type between store and retrieve.

        `default=str` would accept anything and hand back a string where a
        datetime went in, with nothing at read time to reveal the substitution.
        """
        chunk = Chunk(
            page_content="text",
            metadata=ChunkMetadata(
                document=make_document_metadata(extra={"taken": datetime(2024, 3, 5, tzinfo=UTC)}),
                chunk_index=0,
                start_char=0,
                end_char=4,
            ),
        )

        with pytest.raises(ValueError, match="JSON cannot represent"):
            to_record(chunk, make_embedding())

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_metadata_is_rejected(self, bad: float) -> None:
        """json.dumps emits bare NaN and Infinity, which are not JSON.

        They would be stored happily and then fail for anything else reading
        the index, so the promise to reject unrepresentable values has to cover
        them too.
        """
        chunk = Chunk(
            page_content="text",
            metadata=ChunkMetadata(
                document=make_document_metadata(extra={"score": bad}),
                chunk_index=0,
                start_char=0,
                end_char=4,
            ),
        )

        with pytest.raises(ValueError, match="JSON cannot represent"):
            to_record(chunk, make_embedding())

    def test_a_tuple_in_metadata_returns_as_a_list(self) -> None:
        """JSON has one sequence type; this is documented rather than hidden."""
        chunk = Chunk(
            page_content="text",
            metadata=ChunkMetadata(
                document=make_document_metadata(extra={"gps": (50.08, 14.44)}),
                chunk_index=0,
                start_char=0,
                end_char=4,
            ),
        )

        restored = record_to_chunk(to_record(chunk, make_embedding()))

        assert restored.metadata.document.extra == {"gps": [50.08, 14.44]}


class TestStoreLifecycle:
    def test_a_new_store_is_empty(self, store: LanceChunkStore) -> None:
        assert store.count() == 0

    def test_reopening_keeps_the_data(self, tmp_path: Path) -> None:
        """Resumption across process restarts depends on exactly this."""
        path = tmp_path / "index"
        first = LanceChunkStore(path, provenance())
        first.add_document([make_stored_chunk()], [make_embedding()])

        reopened = LanceChunkStore(path, provenance())

        assert reopened.count() == 1

    def test_reopening_with_a_different_width_is_rejected(self, tmp_path: Path) -> None:
        """Otherwise this fails deep inside Arrow, far from the actual cause."""
        path = tmp_path / "index"
        LanceChunkStore(path, provenance())

        with pytest.raises(ValueError, match="embedding model has changed"):
            LanceChunkStore(path, provenance(dimension=DIMENSION + 1))

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_dimension_is_rejected(self, tmp_path: Path, bad: int) -> None:
        with pytest.raises(ValueError, match="dimension must be positive"):
            LanceChunkStore(tmp_path / "index", provenance(dimension=bad))

    def test_repr_identifies_the_table(self, store: LanceChunkStore) -> None:
        assert "chunks" in repr(store)
        assert str(DIMENSION) in repr(store)

    def test_exposes_the_provenance_it_was_opened_with(self, tmp_path: Path) -> None:
        """An indexer needs these to decide what to re-embed."""
        opened = LanceChunkStore(tmp_path / "index", provenance())

        assert opened.provenance == provenance()
        assert opened.provenance.chunk_size == CHUNK_SIZE

    def test_exposes_its_dimension_and_path(self, tmp_path: Path) -> None:
        path = tmp_path / "index"
        opened = LanceChunkStore(path, provenance())

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

        with pytest.raises(ValueError, match="missing columns") as excinfo:
            LanceChunkStore(path, provenance())

        assert "embedding model has changed" not in str(excinfo.value)

    def test_a_table_from_a_different_model_is_rejected(self, tmp_path: Path) -> None:
        """Width is structural compatibility, not identity.

        Two embedders can both emit vectors of this width that occupy entirely
        different spaces. Mixing them passes every structural check, then
        compares incomparable vectors while reporting the older documents as
        already indexed, so they are never re-embedded.
        """
        path = tmp_path / "index"
        LanceChunkStore(path, provenance(embedder_fingerprint="first/model"))

        with pytest.raises(ValueError, match="built with embedder="):
            LanceChunkStore(path, provenance(embedder_fingerprint="second/model"))

    def test_a_table_that_records_no_model_is_rejected(self, tmp_path: Path) -> None:
        """Unverifiable is not the same as compatible."""
        path = tmp_path / "index"
        connection = lancedb.connect(str(path))
        connection.create_table(
            "chunks",
            schema=build_schema(provenance()).remove_metadata(),
        )

        with pytest.raises(ValueError, match="does not record its embedder"):
            LanceChunkStore(path, provenance())

    @pytest.mark.parametrize(
        ("size", "overlap"), [(CHUNK_SIZE + 100, CHUNK_OVERLAP), (CHUNK_SIZE, CHUNK_OVERLAP + 50)]
    )
    def test_a_table_built_with_different_chunking_is_rejected(
        self, tmp_path: Path, size: int, overlap: int
    ) -> None:
        """Chunk settings decide the rows, and never appear in a content hash.

        Without this, changing chunk_size would leave every stored document
        reported as done, so the corpus is silently never re-split and the
        table keeps rows built two different ways.
        """
        path = tmp_path / "index"
        LanceChunkStore(path, provenance())

        with pytest.raises(ValueError, match="must be rebuilt"):
            LanceChunkStore(path, provenance(chunk_size=size, chunk_overlap=overlap))

    def test_unchanged_chunking_reopens_normally(self, tmp_path: Path) -> None:
        """The check must not stand in the way of the resumption it protects."""
        path = tmp_path / "index"
        first = LanceChunkStore(path, provenance())
        first.add_document([make_stored_chunk()], [make_embedding()])

        reopened = LanceChunkStore(path, provenance())

        assert reopened.indexed_content_hashes() == {"a" * 64}

    def test_an_empty_model_id_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="embedder_fingerprint must not be empty"):
            LanceChunkStore(tmp_path / "index", provenance(embedder_fingerprint=""))

    def test_a_column_of_the_wrong_type_is_rejected(self, tmp_path: Path) -> None:
        """Matching names are not compatibility; this would fail on first write."""
        path = tmp_path / "index"
        fields = [
            pa.field("text", pa.int64()) if field.name == "text" else field
            for field in build_schema(provenance())
        ]
        connection = lancedb.connect(str(path))
        connection.create_table(
            "chunks",
            schema=pa.schema(
                fields,
                metadata=build_schema(provenance()).metadata,
            ),
        )

        with pytest.raises(ValueError, match="incompatible columns"):
            LanceChunkStore(path, provenance())

    def test_a_column_with_the_wrong_nullability_is_rejected(self, tmp_path: Path) -> None:
        """Nullability is separate from type in Arrow, and equally fatal.

        A non-null ``page_number`` accepts construction and then rejects the
        None that DOCX and plain text legitimately produce — on the first
        write, which is exactly the late failure this check exists to prevent.
        """
        path = tmp_path / "index"
        expected = build_schema(provenance())
        fields = [
            (
                pa.field(field.name, field.type, nullable=False)
                if field.name == "page_number"
                else field
            )
            for field in expected
        ]
        connection = lancedb.connect(str(path))
        connection.create_table("chunks", schema=pa.schema(fields, metadata=expected.metadata))

        with pytest.raises(ValueError, match="nullable"):
            LanceChunkStore(path, provenance())

    def test_the_recorded_embedder_survives_a_real_reopen(self, tmp_path: Path) -> None:
        """The model check is worthless if LanceDB does not persist the metadata.

        Asserting it on `build_schema` output alone would test PyArrow, not the
        database this store actually depends on.
        """
        path = tmp_path / "index"
        LanceChunkStore(path, provenance(embedder_fingerprint="BAAI/bge-m3"))

        reopened = lancedb.connect(str(path)).open_table("chunks")

        assert reopened.schema.metadata[b"local_rag.embedder_fingerprint"] == b"BAAI/bge-m3"
        assert reopened.schema.metadata[b"local_rag.embedding_dimension"] == str(DIMENSION).encode()

    def test_a_table_with_an_extra_column_is_rejected(self, tmp_path: Path) -> None:
        """The merge writes rows sanitised against the target schema.

        A column this store does not know about is therefore overwritten with
        null, or fails the first write if it is non-nullable. Accepting the
        table would be claiming it is writable when it is not.
        """
        path = tmp_path / "index"
        expected = build_schema(provenance())
        fields = [*expected, pa.field("someone_elses_column", pa.string())]
        connection = lancedb.connect(str(path))
        connection.create_table("chunks", schema=pa.schema(fields, metadata=expected.metadata))

        with pytest.raises(ValueError, match="does not write"):
            LanceChunkStore(path, provenance())

    def test_an_unexpected_open_failure_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A permission or I/O error is not an absent table.

        LanceDB reports both as a plain ValueError, so treating every one as
        "missing" would try to create the table, fail differently, and report a
        cause that has nothing to do with what went wrong.
        """

        class RefusingConnection:
            def open_table(self, name: str) -> object:
                raise ValueError("Permission denied while opening table")

        monkeypatch.setattr(lancedb, "connect", lambda _uri: RefusingConnection())

        with pytest.raises(ValueError, match="Permission denied"):
            LanceChunkStore(tmp_path / "index", provenance())

    def test_a_foreign_table_is_reported_clearly(self, tmp_path: Path) -> None:
        """Pointing at someone else's database should say so, not raise KeyError.

        Without this the schema lookup fails on a missing column and surfaces as
        an unhandled KeyError, which says nothing about what went wrong.
        """
        path = tmp_path / "index"
        connection = lancedb.connect(str(path))
        connection.create_table("chunks", schema=pa.schema([pa.field("x", pa.int32())]))

        with pytest.raises(ValueError, match="was not created by this store"):
            LanceChunkStore(path, provenance())


class TestAdding:
    def test_writes_rows(self, store: LanceChunkStore) -> None:
        written = store.add_document(
            [make_stored_chunk(index=0), make_stored_chunk(index=1)],
            [make_embedding(1.0), make_embedding(2.0)],
        )

        assert written == 2
        assert store.count() == 2

    def test_adding_nothing_is_a_no_op(self, store: LanceChunkStore) -> None:
        assert store.add_document([], []) == 0
        assert store.count() == 0

    def test_an_empty_store_returns_no_records(self, store: LanceChunkStore) -> None:
        assert store.to_records() == []

    def test_rewriting_a_document_replaces_rather_than_duplicates(
        self, store: LanceChunkStore
    ) -> None:
        """LanceDB does not enforce uniqueness, so a derived id alone proves nothing.

        Retrying an interrupted document — the ordinary case for a two-hour
        indexing run — would otherwise double every row it had already written.
        """
        chunks = [make_stored_chunk(index=index) for index in range(3)]
        embeddings = [make_embedding(float(index)) for index in range(3)]
        store.add_document(chunks, embeddings)

        store.add_document(chunks, embeddings)

        assert store.count() == 3

    def test_rewriting_a_document_updates_its_text(self, store: LanceChunkStore) -> None:
        """An upsert must replace the row, not silently keep the older copy."""
        store.add_document([make_stored_chunk("before")], [make_embedding()])

        store.add_document([make_stored_chunk("after")], [make_embedding()])

        assert [row["text"] for row in store.to_records()] == ["after"]

    def test_chunks_from_several_documents_are_rejected(self, store: LanceChunkStore) -> None:
        """One commit per document is what makes a stored hash mean "complete".

        Writing two documents together would let a crash leave one of them
        half-stored while its hash reads as done, and the remainder would be
        skipped for good on the next run.
        """
        with pytest.raises(ValueError, match="one document per call"):
            store.add_document(
                [
                    make_stored_chunk(content_hash="a" * 64),
                    make_stored_chunk(content_hash="b" * 64),
                ],
                [make_embedding(), make_embedding(2.0)],
            )

    def test_a_shorter_rewrite_drops_the_chunks_it_no_longer_has(
        self, store: LanceChunkStore
    ) -> None:
        """An edited document must replace its chunks, not merge into them.

        An upsert alone leaves the trailing chunk of the previous version in
        place: stale text, still retrievable, inside a document that reports
        itself complete.
        """
        store.add_document(
            [make_stored_chunk(f"chunk {index}", index=index) for index in range(3)],
            [make_embedding(float(index)) for index in range(3)],
        )

        store.add_document(
            [make_stored_chunk(f"rewritten {index}", index=index) for index in range(2)],
            [make_embedding(float(index)) for index in range(2)],
        )

        assert store.count() == 2
        assert sorted(row["text"] for row in store.to_records()) == [
            "rewritten 0",
            "rewritten 1",
        ]

    def test_a_shorter_rewrite_leaves_other_documents_alone(self, store: LanceChunkStore) -> None:
        """The deletion is scoped to the document being written."""
        store.add_document(
            [make_stored_chunk(index=index, content_hash="b" * 64) for index in range(2)],
            [make_embedding(float(index)) for index in range(2)],
        )
        store.add_document(
            [make_stored_chunk(index=index) for index in range(3)],
            [make_embedding(float(index)) for index in range(3)],
        )

        store.add_document([make_stored_chunk()], [make_embedding()])

        assert store.indexed_content_hashes() == {"a" * 64, "b" * 64}
        assert store.count() == 3

    def test_repeated_chunk_positions_are_rejected(self, store: LanceChunkStore) -> None:
        """Two chunks at one position derive the same id, which breaks the merge.

        LanceDB documents multiple matches on a merge key as undefined, so this
        could duplicate rows and corrupt every later upsert of the document.
        """
        with pytest.raises(ValueError, match="positions must be unique"):
            store.add_document(
                [make_stored_chunk(index=0), make_stored_chunk(index=0)],
                [make_embedding(), make_embedding(2.0)],
            )

    def test_a_document_is_committed_in_full_or_not_at_all(self, store: LanceChunkStore) -> None:
        """Every chunk lands together, so a present hash is never a partial document."""
        chunks = [make_stored_chunk(index=index) for index in range(5)]
        embeddings = [make_embedding(float(index)) for index in range(5)]

        store.add_document(chunks, embeddings)

        assert store.count() == 5
        assert store.indexed_content_hashes() == {"a" * 64}

    def test_mismatched_lengths_are_rejected(self, store: LanceChunkStore) -> None:
        """Zipping these would pair a chunk with another chunk's vector."""
        with pytest.raises(ValueError, match="correspond one to one"):
            store.add_document([make_stored_chunk()], [])

    def test_a_wrong_width_vector_is_rejected(self, store: LanceChunkStore) -> None:
        wrong = Embedding(dense=(1.0, 2.0))

        with pytest.raises(ValueError, match="store holds 8-dimensional"):
            store.add_document([make_stored_chunk()], [wrong])

    def test_stored_rows_survive_the_round_trip(self, store: LanceChunkStore) -> None:
        """The real test of the schema: through Arrow and back, unchanged."""
        chunk = make_stored_chunk("Smlouva o dilo", index=2, page=3)
        store.add_document([chunk], [make_embedding(5.0)])

        restored = record_to_chunk(store.to_records()[0])

        assert restored == chunk

    def test_vectors_survive_the_round_trip(self, store: LanceChunkStore) -> None:
        embedding = make_embedding(1.5)
        store.add_document([make_stored_chunk()], [embedding])

        restored = record_to_embedding(store.to_records()[0])

        assert restored.dense == pytest.approx(embedding.dense)
        assert restored.sparse is not None
        assert restored.sparse.as_mapping() == {3: 0.5, 7: 0.25}


class TestResumption:
    def test_reports_nothing_when_empty(self, store: LanceChunkStore) -> None:
        assert store.indexed_content_hashes() == set()

    def test_reports_stored_document_hashes(self, store: LanceChunkStore) -> None:
        """Skipping documents already present is what makes a two-hour run survivable."""
        store.add_document([make_stored_chunk(content_hash="a" * 64)], [make_embedding()])
        store.add_document([make_stored_chunk(content_hash="b" * 64)], [make_embedding(2.0)])

        assert store.indexed_content_hashes() == {"a" * 64, "b" * 64}

    def test_reports_every_document_beyond_the_default_query_limit(
        self, store: LanceChunkStore
    ) -> None:
        """LanceDB's query builder defaults to ten rows.

        Relying on that default — or on `limit(0)` meaning unbounded, which has
        not held across releases — would return ten hashes and leave every
        further document re-embedded on the next run, silently and at about six
        seconds a chunk.
        """
        expected = set()
        for index in range(25):
            content_hash = f"{index:064d}"
            expected.add(content_hash)
            store.add_document(
                [make_stored_chunk(content_hash=content_hash)], [make_embedding(float(index))]
            )

        assert store.indexed_content_hashes() == expected

    def test_a_document_appears_once_however_many_chunks_it_has(
        self, store: LanceChunkStore
    ) -> None:
        store.add_document(
            [make_stored_chunk(index=index) for index in range(3)],
            [make_embedding(float(index)) for index in range(3)],
        )

        assert store.indexed_content_hashes() == {"a" * 64}


class TestDeletion:
    def test_deletes_one_document_version(self, store: LanceChunkStore) -> None:
        store.add_document([make_stored_chunk(content_hash="a" * 64)], [make_embedding()])
        store.add_document([make_stored_chunk(content_hash="b" * 64)], [make_embedding(2.0)])

        store.delete_document("a" * 64)

        assert store.indexed_content_hashes() == {"b" * 64}

    def test_deletes_every_chunk_of_that_document(self, store: LanceChunkStore) -> None:
        store.add_document(
            [make_stored_chunk(index=index) for index in range(4)],
            [make_embedding(float(index)) for index in range(4)],
        )

        store.delete_document("a" * 64)

        assert store.count() == 0

    def test_deletes_by_corpus_path(self, store: LanceChunkStore) -> None:
        """An edited file's old chunks must go, whatever version produced them."""
        store.add_document(
            [make_stored_chunk(path="a.pdf", content_hash="a" * 64)], [make_embedding()]
        )
        store.add_document(
            [make_stored_chunk(path="b.pdf", content_hash="b" * 64)], [make_embedding(2.0)]
        )

        store.delete_by_path("a.pdf")

        assert [row["relative_path"] for row in store.to_records()] == ["b.pdf"]

    def test_a_path_containing_an_apostrophe_is_handled(self, store: LanceChunkStore) -> None:
        """Real filenames contain apostrophes, and this builds a SQL predicate."""
        store.add_document(
            [make_stored_chunk(path="Tom's file.pdf", content_hash="a" * 64)], [make_embedding()]
        )
        store.add_document(
            [make_stored_chunk(path="other.pdf", content_hash="b" * 64)], [make_embedding(2.0)]
        )

        store.delete_by_path("Tom's file.pdf")

        assert [row["relative_path"] for row in store.to_records()] == ["other.pdf"]


def test_a_naive_timestamp_from_arrow_is_restored_as_utc() -> None:
    """Arrow can hand back a naive datetime, which DocumentMetadata rejects."""
    record = to_record(make_stored_chunk(), make_embedding())
    record["modified_at"] = datetime(2024, 3, 5, 12, 30)

    assert record_to_chunk(record).metadata.document.modified_at.tzinfo is UTC
