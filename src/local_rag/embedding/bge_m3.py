"""BGE-M3 backend: dense and sparse vectors from one model pass.

BGE-M3 is the reason the interface carries two vectors. A single forward pass
yields a dense representation of meaning and a sparse map of which vocabulary
terms actually fired, so exact-token matching — a contract number, an account
ID, a surname — comes for free rather than requiring a second index to build
and keep consistent.

The model is loaded on first use, not on construction. Building an embedder is
therefore cheap enough to do in configuration wiring, and a process that never
embeds anything never pays for several gigabytes of weights.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from local_rag.embedding.base import DEFAULT_BATCH_SIZE, Embedder, Embedding, SparseVector
from local_rag.errors import MissingDependencyError

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["BgeM3Embedder"]

logger = logging.getLogger(__name__)


class BgeM3Embedder(Embedder):
    """Embeds text with BAAI's BGE-M3 via FlagEmbedding.

    Attributes:
        MODEL_ID: Default HuggingFace identifier.
        DIMENSION: Width of the dense vectors BGE-M3 produces.
        DEFAULT_MAX_LENGTH: Token budget per text. Far below the model's 8192
            ceiling on purpose — see :meth:`__init__`.
    """

    MODEL_ID = "BAAI/bge-m3"
    DIMENSION = 1024
    DEFAULT_MAX_LENGTH = 1024

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | None = None,
        use_fp16: bool | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        """Configure the embedder without loading anything.

        Args:
            model_id: HuggingFace identifier of the model to load.
            batch_size: Texts per forward pass.
            device: ``"cpu"``, ``"cuda"``, or ``None`` to pick automatically.
                Automatic selection prefers CUDA when available; on a card with
                little memory, passing ``"cpu"`` explicitly avoids an
                out-of-memory failure part-way through an indexing run.
            use_fp16: Half precision. Defaults to on for CUDA, which roughly
                halves the memory the weights occupy, and off for CPU, where it
                is generally slower rather than faster.
            max_length: Tokens kept per text; the model truncates beyond this.
                The default is well under BGE-M3's 8192 ceiling because chunks
                arrive at roughly ``chunk_size`` characters — about 350 tokens
                at the configured default — so a larger window would reserve
                memory and compute for capacity that is never used.

        Raises:
            ValueError: If ``batch_size`` or ``max_length`` is not positive.
        """
        super().__init__(batch_size=batch_size)
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")

        self._model_id = model_id
        self._device = device
        self._use_fp16 = use_fp16
        self._max_length = max_length
        self._model: Any | None = None

    @property
    def dimension(self) -> int:
        """Width of the dense vectors this model produces."""
        return self.DIMENSION

    @property
    def supports_sparse(self) -> bool:
        """BGE-M3 returns term weights alongside every dense vector."""
        return True

    @property
    def model_id(self) -> str:
        """Identifier of the model this embedder loads."""
        return self._model_id

    @property
    def max_length(self) -> int:
        """Tokens kept per text before the model truncates."""
        return self._max_length

    @property
    def is_loaded(self) -> bool:
        """Whether the weights have been loaded yet."""
        return self._model is not None

    def _resolve_device(self) -> str:
        """Choose a device, preferring CUDA when it is actually usable."""
        if self._device is not None:
            return self._device

        try:
            import torch  # noqa: PLC0415  # optional dependency, imported on use
        except ImportError:
            return "cpu"

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self) -> Any:
        """Load the weights.

        Separated from :meth:`_embed_batch` so that tests can substitute a stub
        without a multi-gigabyte download, and so the cost is paid once.

        Returns:
            The loaded FlagEmbedding model.

        Raises:
            MissingDependencyError: If FlagEmbedding is not installed.
        """
        try:
            from FlagEmbedding import BGEM3FlagModel  # noqa: PLC0415  # optional dependency
        except ImportError as error:
            raise MissingDependencyError(
                "BGE-M3 requires FlagEmbedding: pip install 'local-rag[embeddings]'"
            ) from error

        device = self._resolve_device()
        use_fp16 = self._use_fp16 if self._use_fp16 is not None else device.startswith("cuda")

        logger.info("Loading %s on %s (fp16=%s)", self._model_id, device, use_fp16)
        return BGEM3FlagModel(self._model_id, use_fp16=use_fp16, devices=device)

    def _ensure_model(self) -> Any:
        """Return the loaded model, loading it on first use."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _embed_batch(self, texts: Sequence[str]) -> list[Embedding]:
        """Run one forward pass and convert its output.

        Args:
            texts: Batch to embed.

        Returns:
            One embedding per text, each carrying dense and sparse vectors.

        Raises:
            RuntimeError: If the model's output is missing the dense or sparse
                component, or the two disagree in length. The base class checks
                counts and widths against what was requested; this checks that
                the payload is shaped the way FlagEmbedding documents at all.
        """
        model = self._ensure_model()
        output = model.encode(
            list(texts),
            batch_size=len(texts),
            max_length=self._max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        try:
            dense_rows = output["dense_vecs"]
            sparse_rows = output["lexical_weights"]
        except (KeyError, TypeError) as error:
            raise RuntimeError(f"unexpected BGE-M3 output shape: {error}") from error

        if len(dense_rows) != len(sparse_rows):
            raise RuntimeError(
                f"BGE-M3 returned {len(dense_rows)} dense and {len(sparse_rows)} sparse "
                f"vectors for the same batch"
            )

        return [
            Embedding(
                dense=tuple(float(value) for value in dense_row),
                sparse=_to_sparse_vector(sparse_row),
            )
            for dense_row, sparse_row in zip(dense_rows, sparse_rows, strict=True)
        ]


def _to_sparse_vector(weights: Any) -> SparseVector:
    """Convert FlagEmbedding's lexical weights into a :class:`SparseVector`.

    The model reports weights as a mapping keyed by token id, with the ids
    rendered as strings and the weights as NumPy scalars. Zero weights do occur
    for tokens that ended up contributing nothing, which is why this builds
    through ``from_mapping`` rather than the strict constructor.

    Args:
        weights: One row of ``lexical_weights``.

    Returns:
        The row as a sparse vector, ordered by token id.

    Raises:
        RuntimeError: If the row is not a mapping of numeric keys to numbers.
    """
    try:
        return SparseVector.from_mapping(
            {int(token): float(weight) for token, weight in weights.items()}
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(f"unexpected BGE-M3 lexical weights: {error}") from error
