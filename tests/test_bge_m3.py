"""Tests for the BGE-M3 backend.

The model itself is never loaded here. A stub stands in for FlagEmbedding so
that the adapter's own logic — lazy loading, device selection, output
conversion — is tested without a multi-gigabyte download, and so CI needs
neither torch nor network access.
"""

from __future__ import annotations

import builtins
import logging
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar

import pytest

from local_rag.embedding import BgeM3Embedder, Embedding
from local_rag.errors import MissingDependencyError


class StubModel:
    """Mimics the FlagEmbedding call shape and records how it was invoked."""

    def __init__(self, dimension: int = BgeM3Embedder.DIMENSION) -> None:
        self.calls: list[dict[str, Any]] = []
        self._dimension = dimension

    def encode(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"texts": list(texts), **kwargs})
        return {
            "dense_vecs": [[float(index + 1)] * self._dimension for index, _ in enumerate(texts)],
            "lexical_weights": [{"7": 0.5, "9": 0.25} for _ in texts],
        }


@pytest.fixture
def stub() -> StubModel:
    """A stub standing in for the loaded model."""
    return StubModel()


@pytest.fixture
def embedder(stub: StubModel, monkeypatch: pytest.MonkeyPatch) -> BgeM3Embedder:
    """An embedder whose weights are replaced by the stub."""
    instance = BgeM3Embedder(batch_size=2)
    monkeypatch.setattr(instance, "_load_model", lambda: stub)
    return instance


class TestCapabilities:
    def test_declares_the_bge_m3_dimension(self) -> None:
        assert BgeM3Embedder().dimension == 1024

    def test_declares_sparse_support(self) -> None:
        """The interface guard requires this to match what is actually returned."""
        assert BgeM3Embedder().supports_sparse is True

    def test_exposes_its_model_id(self) -> None:
        assert BgeM3Embedder().model_id == "BAAI/bge-m3"

    def test_model_id_can_be_overridden(self) -> None:
        assert BgeM3Embedder(model_id="local/copy").model_id == "local/copy"

    def test_the_fingerprint_covers_settings_that_change_the_vectors(self) -> None:
        """max_length decides where the model stops reading, so it changes output.

        An index keyed on the model name alone would report every stored
        document as current and never re-embed it after this changed.
        """
        assert (
            BgeM3Embedder(max_length=512).fingerprint != BgeM3Embedder(max_length=1024).fingerprint
        )
        assert "BAAI/bge-m3" in BgeM3Embedder().fingerprint

    def test_max_length_defaults_well_below_the_model_ceiling(self) -> None:
        """8192 would reserve memory for capacity our chunk size never uses."""
        assert BgeM3Embedder().max_length == 1024

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_max_length_is_rejected(self, bad: int) -> None:
        with pytest.raises(ValueError, match="max_length must be positive"):
            BgeM3Embedder(max_length=bad)


class TestLazyLoading:
    def test_construction_loads_nothing(self) -> None:
        """Building an embedder must be cheap enough for configuration wiring."""
        assert BgeM3Embedder().is_loaded is False

    def test_the_model_loads_on_first_use(self, embedder: BgeM3Embedder) -> None:
        embedder.embed_documents(["a"])

        assert embedder.is_loaded is True

    def test_the_model_is_loaded_only_once(
        self, monkeypatch: pytest.MonkeyPatch, stub: StubModel
    ) -> None:
        instance = BgeM3Embedder(batch_size=1)
        loads = 0

        def count_load() -> StubModel:
            nonlocal loads
            loads += 1
            return stub

        monkeypatch.setattr(instance, "_load_model", count_load)
        instance.embed_documents(["a", "b", "c"])

        assert loads == 1

    def test_embedding_nothing_never_loads_the_model(self, embedder: BgeM3Embedder) -> None:
        """Several gigabytes of weights for zero texts would be a poor trade."""
        assert embedder.embed_documents([]) == []
        assert embedder.is_loaded is False


class TestOutputConversion:
    def test_produces_one_embedding_per_text(self, embedder: BgeM3Embedder) -> None:
        embeddings = embedder.embed_documents(["a", "b", "c"])

        assert len(embeddings) == 3
        assert all(isinstance(item, Embedding) for item in embeddings)

    def test_dense_vectors_have_the_declared_width(self, embedder: BgeM3Embedder) -> None:
        assert embedder.embed_documents(["a"])[0].dimension == BgeM3Embedder.DIMENSION

    def test_sparse_weights_are_converted(self, embedder: BgeM3Embedder) -> None:
        """Token ids arrive as strings and weights as NumPy scalars."""
        sparse = embedder.embed_documents(["a"])[0].sparse

        assert sparse is not None
        assert sparse.as_mapping() == {7: 0.5, 9: 0.25}

    def test_zero_weights_are_dropped(self, embedder: BgeM3Embedder, stub: StubModel) -> None:
        """A term contributing nothing must not occupy a slot in the vector."""

        def with_zero(texts: list[str], **_: Any) -> dict[str, Any]:
            return {
                "dense_vecs": [[1.0] * BgeM3Embedder.DIMENSION for _ in texts],
                "lexical_weights": [{"7": 0.5, "8": 0.0} for _ in texts],
            }

        stub.encode = with_zero  # type: ignore[method-assign]

        sparse = embedder.embed_documents(["a"])[0].sparse

        assert sparse is not None
        assert sparse.as_mapping() == {7: 0.5}

    def test_batches_are_passed_through(self, embedder: BgeM3Embedder, stub: StubModel) -> None:
        embedder.embed_documents(["a", "b", "c"])

        assert [call["texts"] for call in stub.calls] == [["a", "b"], ["c"]]

    def test_the_configured_max_length_reaches_the_model(
        self, monkeypatch: pytest.MonkeyPatch, stub: StubModel
    ) -> None:
        instance = BgeM3Embedder(max_length=256)
        monkeypatch.setattr(instance, "_load_model", lambda: stub)

        instance.embed_documents(["a"])

        assert stub.calls[0]["max_length"] == 256

    def test_colbert_vectors_are_not_requested(
        self, embedder: BgeM3Embedder, stub: StubModel
    ) -> None:
        """Computing them would cost time for output nothing consumes."""
        embedder.embed_documents(["a"])

        assert stub.calls[0]["return_colbert_vecs"] is False
        assert stub.calls[0]["return_dense"] is True
        assert stub.calls[0]["return_sparse"] is True


class TestMalformedModelOutput:
    def test_missing_dense_component_is_reported(
        self, embedder: BgeM3Embedder, stub: StubModel
    ) -> None:
        stub.encode = lambda texts, **_: {"lexical_weights": [{} for _ in texts]}  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="unexpected BGE-M3 output shape"):
            embedder.embed_documents(["a"])

    def test_missing_sparse_component_is_reported(
        self, embedder: BgeM3Embedder, stub: StubModel
    ) -> None:
        stub.encode = lambda texts, **_: {"dense_vecs": [[1.0] for _ in texts]}  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="unexpected BGE-M3 output shape"):
            embedder.embed_documents(["a"])

    def test_mismatched_component_lengths_are_reported(
        self, embedder: BgeM3Embedder, stub: StubModel
    ) -> None:
        """Zipping these blindly would pair a vector with another text's terms."""

        def lopsided(texts: list[str], **_: Any) -> dict[str, Any]:
            return {
                "dense_vecs": [[1.0] * BgeM3Embedder.DIMENSION for _ in texts],
                "lexical_weights": [{"7": 0.5}],
            }

        stub.encode = lopsided  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="2 dense and 1 sparse"):
            embedder.embed_documents(["a", "b"])

    def test_unusable_lexical_weights_are_reported(
        self, embedder: BgeM3Embedder, stub: StubModel
    ) -> None:
        def not_a_mapping(texts: list[str], **_: Any) -> dict[str, Any]:
            return {
                "dense_vecs": [[1.0] * BgeM3Embedder.DIMENSION for _ in texts],
                "lexical_weights": ["not a mapping" for _ in texts],
            }

        stub.encode = not_a_mapping  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="unexpected BGE-M3 lexical weights"):
            embedder.embed_documents(["a"])


def fake_torch(*, cuda_available: bool) -> ModuleType:
    """A stand-in torch module exposing only what device selection consults."""
    module = ModuleType("torch")
    module.cuda = SimpleNamespace(is_available=lambda: cuda_available)  # type: ignore[attr-defined]
    return module


class RecordingFlagModel:
    """Captures the arguments BGEM3FlagModel would have been constructed with.

    The signature deliberately mirrors FlagEmbedding 1.3 exactly rather than
    absorbing ``**kwargs``. A permissive stub accepts any call and would have
    hidden the fact that 1.2 named this argument ``device``; pinning the
    parameter names here means changing the call site breaks the test rather
    than only breaking at runtime against the real library.
    """

    last: ClassVar[dict[str, Any]] = {}

    def __init__(self, model_id: str, *, use_fp16: bool, devices: str) -> None:
        type(self).last = {"model_id": model_id, "use_fp16": use_fp16, "devices": devices}


def fake_flag_embedding() -> ModuleType:
    """A stand-in FlagEmbedding module exporting the recording model."""
    module = ModuleType("FlagEmbedding")
    module.BGEM3FlagModel = RecordingFlagModel  # type: ignore[attr-defined]
    return module


class TestDeviceSelection:
    def test_an_explicit_device_is_respected(self) -> None:
        assert BgeM3Embedder(device="cpu")._resolve_device() == "cpu"

    def test_prefers_cuda_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda_available=True))

        assert BgeM3Embedder()._resolve_device() == "cuda"

    def test_falls_back_to_cpu_without_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda_available=False))

        assert BgeM3Embedder()._resolve_device() == "cpu"

    def test_an_explicit_device_wins_over_available_cuda(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The escape hatch for cards too small to hold the model."""
        monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda_available=True))

        assert BgeM3Embedder(device="cpu")._resolve_device() == "cpu"


class TestModelConstruction:
    """Covers the load path itself, with FlagEmbedding replaced by a recorder."""

    @pytest.fixture(autouse=True)
    def _install_stub_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "FlagEmbedding", fake_flag_embedding())
        RecordingFlagModel.last = {}

    def test_passes_the_model_id_and_resolved_device(self) -> None:
        BgeM3Embedder(device="cpu")._load_model()

        assert RecordingFlagModel.last["model_id"] == "BAAI/bge-m3"
        assert RecordingFlagModel.last["devices"] == "cpu"

    def test_fp16_is_off_on_cpu(self) -> None:
        """Half precision is generally slower than float32 on CPU, not faster."""
        BgeM3Embedder(device="cpu")._load_model()

        assert RecordingFlagModel.last["use_fp16"] is False

    def test_fp16_is_on_for_cuda(self) -> None:
        """It roughly halves the memory the weights occupy."""
        BgeM3Embedder(device="cuda")._load_model()

        assert RecordingFlagModel.last["use_fp16"] is True

    @pytest.mark.parametrize("requested", [True, False])
    def test_an_explicit_fp16_choice_overrides_the_device_default(self, requested: bool) -> None:
        BgeM3Embedder(device="cpu", use_fp16=requested)._load_model()

        assert RecordingFlagModel.last["use_fp16"] is requested

    def test_loading_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """An operation this slow should say that it started."""
        with caplog.at_level(logging.INFO):
            BgeM3Embedder(device="cpu")._load_model()

        assert "BAAI/bge-m3" in caplog.text

    def test_falls_back_to_cpu_without_torch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Device selection must not itself require the optional dependency."""
        real_import = builtins.__import__

        def refuse_torch(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse_torch)

        assert BgeM3Embedder()._resolve_device() == "cpu"


class TestMissingDependency:
    def test_loading_without_flagembedding_names_the_extra(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The error must say how to fix it, not merely that an import failed."""
        real_import = builtins.__import__

        def refuse_flagembedding(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "FlagEmbedding":
                raise ModuleNotFoundError("no FlagEmbedding", name="FlagEmbedding")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse_flagembedding)

        with (
            caplog.at_level(logging.INFO),
            pytest.raises(MissingDependencyError, match=r"local-rag\[embeddings\]"),
        ):
            BgeM3Embedder(device="cpu").embed_documents(["a"])

    def test_a_broken_dependency_is_not_reported_as_a_missing_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An installed FlagEmbedding failing its own imports must surface as itself.

        Reporting it as "FlagEmbedding is not installed" would send someone to
        reinstall an extra that is already present, hiding the transitive
        module that actually went missing.
        """
        real_import = builtins.__import__

        def break_inside_flagembedding(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "FlagEmbedding":
                raise ModuleNotFoundError("No module named 'peft'", name="peft")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", break_inside_flagembedding)

        with pytest.raises(ModuleNotFoundError, match="peft") as excinfo:
            BgeM3Embedder(device="cpu").embed_documents(["a"])

        assert not isinstance(excinfo.value, MissingDependencyError)

    def test_a_submodule_of_the_extra_still_counts_as_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`FlagEmbedding.inference` being absent still means the extra is unusable."""
        real_import = builtins.__import__

        def refuse_submodule(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "FlagEmbedding":
                raise ModuleNotFoundError("gone", name="FlagEmbedding.inference")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse_submodule)

        with pytest.raises(MissingDependencyError, match=r"local-rag\[embeddings\]"):
            BgeM3Embedder(device="cpu").embed_documents(["a"])
