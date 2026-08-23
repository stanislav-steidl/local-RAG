"""Tests for the embedding abstraction: vector types, batching and guards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from local_rag.embedding import DEFAULT_BATCH_SIZE, Embedder, Embedding, SparseVector

if TYPE_CHECKING:
    from collections.abc import Sequence


class RecordingEmbedder(Embedder):
    """A deterministic embedder that records the batches it was handed.

    Deterministic so assertions can be exact, and recording so batching itself
    is observable rather than inferred.
    """

    def __init__(self, *, batch_size: int = DEFAULT_BATCH_SIZE, width: int = 3) -> None:
        super().__init__(batch_size=batch_size)
        self.calls: list[list[str]] = []
        self._width = width

    @property
    def dimension(self) -> int:
        return self._width

    def _embed_batch(self, texts: Sequence[str]) -> list[Embedding]:
        self.calls.append(list(texts))
        return [
            Embedding(dense=tuple(float(len(text)) + offset for offset in range(self._width)))
            for text in texts
        ]


class MiscountingEmbedder(RecordingEmbedder):
    """Returns fewer embeddings than it was asked for."""

    def _embed_batch(self, texts: Sequence[str]) -> list[Embedding]:
        return super()._embed_batch(texts)[:-1]


class InconsistentWidthEmbedder(RecordingEmbedder):
    """Returns dense vectors of differing widths."""

    def _embed_batch(self, texts: Sequence[str]) -> list[Embedding]:
        return [
            Embedding(dense=tuple(1.0 for _ in range(index + 1))) for index, _ in enumerate(texts)
        ]


class TestSparseVector:
    def test_defaults_to_empty(self) -> None:
        vector = SparseVector()
        assert len(vector) == 0
        assert vector.as_mapping() == {}

    def test_exposes_terms_as_a_mapping(self) -> None:
        vector = SparseVector(indices=(4, 9), values=(0.5, 0.25))
        assert vector.as_mapping() == {4: 0.5, 9: 0.25}

    def test_length_counts_non_zero_terms(self) -> None:
        assert len(SparseVector(indices=(1, 2, 3), values=(0.1, 0.2, 0.3))) == 3

    def test_mismatched_array_lengths_are_rejected(self) -> None:
        """Parallel arrays that disagree would silently mispair term weights."""
        with pytest.raises(ValueError, match="parallel"):
            SparseVector(indices=(1, 2), values=(0.5,))

    def test_negative_indices_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            SparseVector(indices=(-1,), values=(0.5,))

    def test_duplicate_indices_are_rejected(self) -> None:
        """A repeated term has an ambiguous weight."""
        with pytest.raises(ValueError, match="unique"):
            SparseVector(indices=(3, 3), values=(0.5, 0.25))


class TestEmbedding:
    def test_reports_its_dimension(self) -> None:
        assert Embedding(dense=(1.0, 2.0, 3.0)).dimension == 3

    def test_sparse_defaults_to_absent(self) -> None:
        """Absent is distinct from present-but-empty: it means dense-only."""
        assert Embedding(dense=(1.0,)).sparse is None

    def test_carries_a_sparse_vector_when_given_one(self) -> None:
        sparse = SparseVector(indices=(2,), values=(0.75,))
        assert Embedding(dense=(1.0,), sparse=sparse).sparse is sparse

    def test_empty_dense_vector_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            Embedding(dense=())

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_values_are_rejected(self, bad: float) -> None:
        """NaN propagates through cosine similarity and poisons every ranking."""
        with pytest.raises(ValueError, match="NaN or infinity"):
            Embedding(dense=(1.0, bad))


class TestBatching:
    def test_splits_input_into_batches(self) -> None:
        embedder = RecordingEmbedder(batch_size=2)

        embedder.embed_documents(["a", "b", "c", "d", "e"])

        assert embedder.calls == [["a", "b"], ["c", "d"], ["e"]]

    def test_returns_one_embedding_per_text_in_order(self) -> None:
        embedder = RecordingEmbedder(batch_size=2)

        embeddings = embedder.embed_documents(["a", "bb", "ccc"])

        assert [embedding.dense[0] for embedding in embeddings] == [1.0, 2.0, 3.0]

    def test_a_single_batch_is_not_split(self) -> None:
        embedder = RecordingEmbedder(batch_size=10)

        embedder.embed_documents(["a", "b"])

        assert embedder.calls == [["a", "b"]]

    def test_empty_input_never_calls_the_backend(self) -> None:
        """Loading a model to embed nothing would be an expensive no-op."""
        embedder = RecordingEmbedder()

        assert embedder.embed_documents([]) == []
        assert embedder.calls == []

    def test_batch_size_is_exposed(self) -> None:
        assert RecordingEmbedder(batch_size=4).batch_size == 4

    @pytest.mark.parametrize("size", [0, -1])
    def test_non_positive_batch_size_is_rejected(self, size: int) -> None:
        with pytest.raises(ValueError, match="batch_size must be positive"):
            RecordingEmbedder(batch_size=size)


class TestBackendGuards:
    def test_a_miscounting_backend_is_caught(self) -> None:
        """Silently dropping one embedding misaligns every chunk after it.

        The symptom would appear much later, as retrieval confidently returning
        the wrong document, so it has to be caught at the source.
        """
        embedder = MiscountingEmbedder(batch_size=3)

        with pytest.raises(RuntimeError, match="returned 2 embeddings for 3 texts"):
            embedder.embed_documents(["a", "b", "c"])

    def test_inconsistent_dense_widths_are_caught(self) -> None:
        """A vector store cannot hold rows of differing width."""
        embedder = InconsistentWidthEmbedder(batch_size=3)

        with pytest.raises(RuntimeError, match="mixed dense widths"):
            embedder.embed_documents(["a", "b", "c"])


class TestQueryEmbedding:
    def test_embeds_a_single_query(self) -> None:
        embedder = RecordingEmbedder()

        assert embedder.embed_query("hello").dense[0] == 5.0

    def test_query_goes_through_the_backend_once(self) -> None:
        embedder = RecordingEmbedder()

        embedder.embed_query("hello")

        assert embedder.calls == [["hello"]]


class TestCapabilities:
    def test_dense_only_by_default(self) -> None:
        """Backends opt in to sparse rather than claiming it by default."""
        assert RecordingEmbedder().supports_sparse is False

    def test_dimension_is_reported(self) -> None:
        assert RecordingEmbedder(width=5).dimension == 5

    def test_interface_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            Embedder()  # type: ignore[abstract]
