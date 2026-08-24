"""Integration tests exercising the real BGE-M3 weights.

Every other test in this suite substitutes a stub, which verifies the adapter
against assumptions rather than against the library. That gap is not
theoretical: a wrong dependency floor once passed a fully covered suite because
nothing had executed the real API. These tests close it.

They are marked ``slow`` and excluded from CI, which would otherwise download
2.3 GB of weights per job. Run them deliberately::

    pytest -m slow

Assertions are deliberately loose about exact scores, which shift between model
revisions, and strict about the properties the pipeline actually depends on:
vector shape, sparse structure, and whether the right document ranks first.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from local_rag.embedding import Embedding

pytest.importorskip("FlagEmbedding", reason="needs the embeddings extra")

from local_rag.embedding import BgeM3Embedder

pytestmark = [pytest.mark.slow, pytest.mark.requires_embeddings]

#: Short stand-ins for the kinds of document the corpus holds. Synthetic, with
#: the diacritics a wrong codepage or tokenizer would destroy.
DOCUMENTS = {
    "contract": "Smlouva o dílo mezi objednatelem a zhotovitelem, cena 250 000 Kč bez DPH.",
    "invoice": "Faktura č. 2024/0731, splatnost 14 dní, částka 12 345 Kč.",
    "power_of_attorney": "Plná moc k zastupování ve věci převodu nemovitosti.",
    "employment": "Employment agreement between the company and the employee.",
}


def cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Cosine similarity between two dense vectors."""
    dot = sum(x * y for x, y in zip(left, right, strict=True))
    norm = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
    return dot / norm if norm else 0.0


@pytest.fixture(scope="module")
def embedder() -> BgeM3Embedder:
    """A CPU embedder, loaded once for the whole module.

    CPU rather than automatic: the suite must behave identically on a machine
    whose GPU cannot hold the model.
    """
    return BgeM3Embedder(device="cpu", batch_size=4)


@pytest.fixture(scope="module")
def corpus(embedder: BgeM3Embedder) -> dict[str, Embedding]:
    """The sample documents, embedded once."""
    keys = list(DOCUMENTS)
    vectors = embedder.embed_documents([DOCUMENTS[key] for key in keys])
    return dict(zip(keys, vectors, strict=True))


class TestRealVectorShape:
    def test_dense_width_matches_the_declared_dimension(
        self, embedder: BgeM3Embedder, corpus: dict[str, Embedding]
    ) -> None:
        """The interface guard trusts `dimension`; this confirms it is truthful."""
        assert all(vector.dimension == embedder.dimension for vector in corpus.values())

    def test_dense_vectors_are_normalised(self, corpus: dict[str, Embedding]) -> None:
        """BGE-M3 returns unit vectors, so cosine reduces to a dot product."""
        for vector in corpus.values():
            assert math.isclose(
                math.sqrt(sum(value * value for value in vector.dense)), 1.0, abs_tol=1e-3
            )

    def test_every_document_carries_sparse_terms(self, corpus: dict[str, Embedding]) -> None:
        for vector in corpus.values():
            assert vector.sparse is not None
            assert len(vector.sparse) > 0

    def test_sparse_terms_have_the_types_the_store_expects(
        self, corpus: dict[str, Embedding]
    ) -> None:
        """Token ids arrive from the model as strings and weights as NumPy scalars."""
        sparse = corpus["invoice"].sparse
        assert sparse is not None
        assert all(isinstance(index, int) for index in sparse.indices)
        assert all(isinstance(value, float) for value in sparse.values)
        assert all(value != 0.0 for value in sparse.values)


class TestRealRetrieval:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("Kde mám fakturu na 12 345 korun?", "invoice"),
            ("plná moc nemovitost", "power_of_attorney"),
            ("employment contract with the employer", "employment"),
            ("smlouva o dílo cena bez DPH", "contract"),
        ],
    )
    def test_the_intended_document_ranks_first(
        self,
        embedder: BgeM3Embedder,
        corpus: dict[str, Embedding],
        query: str,
        expected: str,
    ) -> None:
        """The end-to-end claim: a natural question finds the right document."""
        embedded = embedder.embed_query(query)
        ranked = sorted(
            corpus, key=lambda key: cosine(embedded.dense, corpus[key].dense), reverse=True
        )

        assert ranked[0] == expected

    def test_retrieval_crosses_languages(
        self, embedder: BgeM3Embedder, corpus: dict[str, Embedding]
    ) -> None:
        """An English query should still reach a Czech document about the same thing.

        This is the property that justified a multilingual model over an
        English-only one for a bilingual corpus.
        """
        embedded = embedder.embed_query("invoice payable in fourteen days")

        assert cosine(embedded.dense, corpus["invoice"].dense) > cosine(
            embedded.dense, corpus["power_of_attorney"].dense
        )


class TestRealSparseMatching:
    def test_an_exact_token_separates_documents_that_contain_it(
        self, embedder: BgeM3Embedder, corpus: dict[str, Embedding]
    ) -> None:
        """The reason for choosing a model that emits sparse vectors at all.

        Searching a document number is the archetypal archive query, and it is
        what dense similarity handles least reliably.
        """
        query = embedder.embed_query("2024/0731")
        assert query.sparse is not None
        query_terms = set(query.sparse.indices)

        def overlap(key: str) -> int:
            sparse = corpus[key].sparse
            assert sparse is not None
            return len(query_terms & set(sparse.indices))

        assert overlap("invoice") > 0
        assert overlap("contract") == 0
        assert overlap("power_of_attorney") == 0
