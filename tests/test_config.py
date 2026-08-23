"""Tests for runtime configuration loading and validation.

Every test runs in an isolated working directory with all ``LOCAL_RAG_*``
variables stripped, so a developer's own ``.env`` or exported settings cannot
influence the result.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from local_rag.config import ENV_PREFIX, Settings, get_settings


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip inherited settings and run from a scratch directory."""
    for key in list(os.environ):
        if key.startswith(ENV_PREFIX):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """An existing, empty corpus directory."""
    path = tmp_path / "corpus"
    path.mkdir()
    return path


class TestRequiredFields:
    def test_corpus_dir_is_required(self) -> None:
        """No default may exist, or a private path could be baked into the source."""
        with pytest.raises(ValidationError, match="corpus_dir"):
            Settings()  # type: ignore[call-arg]

    def test_corpus_dir_must_exist(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="does not exist"):
            Settings(corpus_dir=tmp_path / "missing")

    def test_corpus_dir_must_be_a_directory(self, tmp_path: Path) -> None:
        file_path = tmp_path / "notadir.txt"
        file_path.write_text("x", encoding="utf-8")
        with pytest.raises(ValidationError, match="not a directory"):
            Settings(corpus_dir=file_path)


class TestDefaults:
    def test_defaults_are_applied(self, corpus: Path) -> None:
        settings = Settings(corpus_dir=corpus)
        assert settings.embedding_model == "BAAI/bge-m3"
        assert settings.chunk_size == 1200
        assert settings.chunk_overlap == 200
        assert settings.llm_model == "qwen2.5:7b-instruct-q4_K_M"
        assert settings.log_level == "INFO"

    def test_index_dir_defaults_outside_the_corpus(self, corpus: Path) -> None:
        settings = Settings(corpus_dir=corpus)
        assert not settings.index_dir.is_relative_to(settings.corpus_dir)


class TestPathHandling:
    def test_paths_are_resolved_to_absolute(self, corpus: Path, tmp_path: Path) -> None:
        """Behaviour must not depend on the process working directory."""
        settings = Settings(corpus_dir=Path("corpus"), index_dir=Path("idx"))
        assert settings.corpus_dir == corpus
        assert settings.index_dir == tmp_path / "idx"

    def test_user_home_is_expanded(self, monkeypatch: pytest.MonkeyPatch, corpus: Path) -> None:
        monkeypatch.setenv("HOME", str(corpus.parent))
        monkeypatch.setenv("USERPROFILE", str(corpus.parent))
        settings = Settings(corpus_dir=Path("~/corpus"))
        assert settings.corpus_dir == corpus


class TestChunkingInvariants:
    def test_overlap_equal_to_chunk_size_is_rejected(self, corpus: Path) -> None:
        """Equal values would leave the chunker unable to advance."""
        with pytest.raises(ValidationError, match="must be smaller than"):
            Settings(corpus_dir=corpus, chunk_size=500, chunk_overlap=500)

    def test_overlap_larger_than_chunk_size_is_rejected(self, corpus: Path) -> None:
        with pytest.raises(ValidationError, match="must be smaller than"):
            Settings(corpus_dir=corpus, chunk_size=500, chunk_overlap=800)

    def test_zero_overlap_is_allowed(self, corpus: Path) -> None:
        assert Settings(corpus_dir=corpus, chunk_overlap=0).chunk_overlap == 0

    def test_chunk_size_must_be_positive(self, corpus: Path) -> None:
        with pytest.raises(ValidationError, match="chunk_size"):
            Settings(corpus_dir=corpus, chunk_size=0)

    def test_negative_overlap_is_rejected(self, corpus: Path) -> None:
        with pytest.raises(ValidationError, match="chunk_overlap"):
            Settings(corpus_dir=corpus, chunk_overlap=-1)


class TestIndexLocation:
    def test_index_inside_corpus_is_rejected(self, corpus: Path) -> None:
        """An index inside the corpus would be re-indexed and pollute the source folder."""
        with pytest.raises(ValidationError, match="must not be inside"):
            Settings(corpus_dir=corpus, index_dir=corpus / "index")

    def test_index_equal_to_corpus_is_rejected(self, corpus: Path) -> None:
        with pytest.raises(ValidationError, match="must not be inside"):
            Settings(corpus_dir=corpus, index_dir=corpus)

    def test_index_as_a_sibling_is_allowed(self, corpus: Path, tmp_path: Path) -> None:
        settings = Settings(corpus_dir=corpus, index_dir=tmp_path / "index")
        assert settings.index_dir == tmp_path / "index"


class TestEnvironmentLoading:
    def test_settings_are_read_from_prefixed_variables(
        self, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(f"{ENV_PREFIX}CORPUS_DIR", str(corpus))
        monkeypatch.setenv(f"{ENV_PREFIX}CHUNK_SIZE", "800")
        monkeypatch.setenv(f"{ENV_PREFIX}LOG_LEVEL", "DEBUG")

        settings = Settings()  # type: ignore[call-arg]

        assert settings.corpus_dir == corpus
        assert settings.chunk_size == 800
        assert settings.log_level == "DEBUG"

    def test_settings_are_read_from_a_dotenv_file(self, corpus: Path, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text(
            f"{ENV_PREFIX}CORPUS_DIR={corpus}\n{ENV_PREFIX}CHUNK_SIZE=640\n",
            encoding="utf-8",
        )

        settings = Settings()  # type: ignore[call-arg]

        assert settings.corpus_dir == corpus
        assert settings.chunk_size == 640

    def test_environment_overrides_the_dotenv_file(
        self, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / ".env").write_text(
            f"{ENV_PREFIX}CORPUS_DIR={corpus}\n{ENV_PREFIX}CHUNK_SIZE=640\n",
            encoding="utf-8",
        )
        monkeypatch.setenv(f"{ENV_PREFIX}CHUNK_SIZE", "999")

        assert Settings().chunk_size == 999  # type: ignore[call-arg]

    def test_unknown_prefixed_variables_are_rejected(
        self, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A typo must fail loudly instead of silently leaving a default in place."""
        monkeypatch.setenv(f"{ENV_PREFIX}CORPUS_DIR", str(corpus))
        monkeypatch.setenv(f"{ENV_PREFIX}CHUNKSIZE", "800")

        with pytest.raises(ValidationError, match="CHUNKSIZE"):
            Settings()  # type: ignore[call-arg]

    def test_unknown_keys_in_a_dotenv_file_are_rejected(self, corpus: Path, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text(
            f"{ENV_PREFIX}CORPUS_DIR={corpus}\n{ENV_PREFIX}CHUKN_SIZE=640\n",
            encoding="utf-8",
        )

        with pytest.raises(ValidationError, match=r"[Ee]xtra"):
            Settings()  # type: ignore[call-arg]

    def test_unrelated_environment_variables_are_ignored(
        self, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the project's own prefix is policed."""
        monkeypatch.setenv(f"{ENV_PREFIX}CORPUS_DIR", str(corpus))
        monkeypatch.setenv("UNRELATED_CHUNK_SIZE", "800")

        assert Settings().chunk_size == 1200  # type: ignore[call-arg]

    def test_invalid_log_level_is_rejected(
        self, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(f"{ENV_PREFIX}CORPUS_DIR", str(corpus))
        monkeypatch.setenv(f"{ENV_PREFIX}LOG_LEVEL", "VERBOSE")

        with pytest.raises(ValidationError, match="log_level"):
            Settings()  # type: ignore[call-arg]


class TestGetSettings:
    def test_returns_the_same_instance_on_repeated_calls(
        self, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Configuration is read and validated exactly once per process."""
        monkeypatch.setenv(f"{ENV_PREFIX}CORPUS_DIR", str(corpus))
        assert get_settings() is get_settings()
