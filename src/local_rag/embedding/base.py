"""The embedding abstraction: vector types and the interface backends implement.

Embedding is the one pipeline stage with a genuinely heavyweight dependency —
PyTorch, and a model measured in gigabytes. Defining the contract separately
from any backend is what keeps the rest of the pipeline testable without either:
retrieval, indexing and the CLI depend on :class:`Embedder`, and a test can
substitute a deterministic fake.

Two vectors, not one. BGE-M3 emits a dense vector capturing meaning and a
sparse one capturing which terms actually occur. Dense retrieval alone
reliably misses exact tokens — a contract number, an account ID, a surname —
which is precisely what someone searching a document archive tends to type.
Carrying both from the outset is what makes hybrid search possible later
without reshaping the interface.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = ["DEFAULT_BATCH_SIZE", "Embedder", "Embedding", "SparseVector"]

#: Texts embedded per backend call. Small because the target machine is
#: CPU-bound; the value is a constructor argument for hardware that is not.
DEFAULT_BATCH_SIZE = 8


@dataclass(frozen=True, slots=True)
class SparseVector:
    """Term weights for lexical matching, stored as parallel index/value arrays.

    Only non-zero entries are held: a sparse vector over a 250k-token vocabulary
    typically has a few dozen. The arrays are parallel rather than a mapping so
    they translate directly into the columnar layout the vector store expects.

    Attributes:
        indices: Vocabulary positions with a non-zero weight.
        values: The weight at each corresponding index.
    """

    indices: tuple[int, ...] = ()
    values: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Validate that the two arrays describe a coherent vector."""
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"indices and values must be parallel, got {len(self.indices)} and "
                f"{len(self.values)}"
            )
        if any(index < 0 for index in self.indices):
            raise ValueError("indices must be non-negative")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("indices must be unique")

    def __len__(self) -> int:
        """Number of non-zero terms."""
        return len(self.indices)

    def as_mapping(self) -> dict[int, float]:
        """Return the vector as an index-to-weight mapping."""
        return dict(zip(self.indices, self.values, strict=True))


@dataclass(frozen=True, slots=True)
class Embedding:
    """One text's vector representation.

    Attributes:
        dense: The semantic vector.
        sparse: Term weights, when the backend produces them. ``None`` means
            the backend is dense-only, which is different from a sparse vector
            that happens to be empty.
    """

    dense: tuple[float, ...]
    sparse: SparseVector | None = None

    def __post_init__(self) -> None:
        """Validate that the dense vector is usable for similarity search."""
        if not self.dense:
            raise ValueError("dense vector must not be empty")
        if any(not math.isfinite(value) for value in self.dense):
            raise ValueError("dense vector must not contain NaN or infinity")

    @property
    def dimension(self) -> int:
        """Length of the dense vector."""
        return len(self.dense)


class Embedder(ABC):
    """Turns text into vectors.

    Subclasses implement :meth:`_embed_batch` for a single backend call; this
    class handles batching, empty input, and verifying that the backend
    returned what it was asked for.

    ``embed_documents`` and ``embed_query`` mirror LangChain's ``Embeddings``
    interface so that adopting it later is an adapter rather than a rewrite.
    They exist as separate methods even where a model treats both identically —
    BGE-M3 does — because other models do not: the E5 family, for instance,
    requires ``query:`` and ``passage:`` prefixes, and a caller must not have to
    know which kind of model sits behind the interface.
    """

    def __init__(self, *, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        """Configure batching.

        Args:
            batch_size: Texts passed to the backend per call.

        Raises:
            ValueError: If ``batch_size`` is not positive.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self._batch_size = batch_size

    @property
    def batch_size(self) -> int:
        """Texts passed to the backend per call."""
        return self._batch_size

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of the dense vectors this embedder produces."""

    @property
    def supports_sparse(self) -> bool:
        """Whether this embedder also produces term weights."""
        return False

    @abstractmethod
    def _embed_batch(self, texts: Sequence[str]) -> list[Embedding]:
        """Embed one batch.

        Args:
            texts: Non-empty batch, no larger than :attr:`batch_size`.

        Returns:
            One embedding per input text, in the same order.
        """

    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        """Embed texts for storage in the index.

        Args:
            texts: Texts to embed. May be empty.

        Returns:
            One embedding per input text, in input order.

        Raises:
            RuntimeError: If the backend returns the wrong number of
                embeddings, or vectors of inconsistent width. Either would
                otherwise silently misalign embeddings with their chunks, which
                surfaces much later as retrieval that returns the wrong
                document.
        """
        embeddings: list[Embedding] = []
        for batch in self._batches(texts):
            produced = self._embed_batch(batch)
            if len(produced) != len(batch):
                raise RuntimeError(
                    f"backend returned {len(produced)} embeddings for {len(batch)} texts"
                )
            embeddings.extend(produced)

        widths = {embedding.dimension for embedding in embeddings}
        if len(widths) > 1:
            raise RuntimeError(f"backend returned mixed dense widths: {sorted(widths)}")

        return embeddings

    def embed_query(self, text: str) -> Embedding:
        """Embed a single search query.

        Args:
            text: The query.

        Returns:
            Its embedding.
        """
        return self.embed_documents([text])[0]

    def _batches(self, texts: Sequence[str]) -> Iterator[Sequence[str]]:
        """Split ``texts`` into slices of at most :attr:`batch_size`."""
        for offset in range(0, len(texts), self._batch_size):
            yield texts[offset : offset + self._batch_size]
