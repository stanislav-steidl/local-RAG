"""Runtime configuration, resolved from environment variables or a ``.env`` file.

Configuration is the one place where input genuinely originates outside the
program, so unlike :mod:`local_rag.models` it is validated with Pydantic and
fails loudly at startup rather than deep inside an indexing run.

The corpus location is deliberately *not* a default. It must be supplied by the
operator, which keeps any private path out of the source tree entirely.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["LogLevel", "Settings", "get_settings"]

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

ENV_PREFIX = "LOCAL_RAG_"


class Settings(BaseSettings):
    """Validated runtime settings for the whole pipeline.

    Every field may be supplied as an environment variable prefixed with
    ``LOCAL_RAG_`` (for example ``LOCAL_RAG_CHUNK_SIZE``) or as a line in a
    ``.env`` file. Unknown ``LOCAL_RAG_*`` keys are rejected from both sources
    so a typo surfaces immediately instead of silently leaving a default in
    place — ``extra="forbid"`` covers the ``.env`` file, and
    :meth:`_reject_unknown_environment_variables` covers the environment, which
    Pydantic would otherwise ignore.

    Attributes:
        corpus_dir: Directory holding the documents to index. Required.
        index_dir: Where the LanceDB index is written.
        embedding_model: HuggingFace identifier of the embedding model.
        chunk_size: Target chunk length, in characters.
        chunk_overlap: Characters shared between consecutive chunks, so a
            passage split across a boundary remains retrievable.
        llm_model: Ollama model used for answer generation.
        log_level: Logging verbosity.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )

    corpus_dir: Path
    index_dir: Path = Path(".lancedb")
    embedding_model: str = "BAAI/bge-m3"
    chunk_size: int = Field(default=1200, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    llm_model: str = "qwen2.5:7b-instruct-q4_K_M"
    log_level: LogLevel = "INFO"

    @field_validator("corpus_dir", "index_dir")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        """Make paths absolute so behaviour does not depend on the working directory."""
        return value.expanduser().resolve()

    @field_validator("corpus_dir")
    @classmethod
    def _corpus_must_be_an_existing_directory(cls, value: Path) -> Path:
        """Fail at startup rather than part-way through an indexing run."""
        if not value.exists():
            raise ValueError(f"corpus directory does not exist: {value}")
        if not value.is_dir():
            raise ValueError(f"corpus path is not a directory: {value}")
        return value

    @model_validator(mode="after")
    def _overlap_must_be_smaller_than_chunk(self) -> Settings:
        """An overlap at least as large as the chunk would never advance the cursor."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )
        return self

    @model_validator(mode="after")
    def _reject_unknown_environment_variables(self) -> Settings:
        """Reject ``LOCAL_RAG_*`` variables that match no field.

        Pydantic silently ignores unrecognised environment variables, so
        ``LOCAL_RAG_CHUNKSIZE`` would quietly leave ``chunk_size`` at its
        default. For a pipeline whose retrieval quality depends on these
        values, failing to apply a setting must not be silent.
        """
        known = {f"{ENV_PREFIX}{name.upper()}" for name in type(self).model_fields}
        unknown = sorted(
            key for key in os.environ if key.startswith(ENV_PREFIX) and key.upper() not in known
        )
        if unknown:
            raise ValueError(f"unknown {ENV_PREFIX}* environment variable(s): {', '.join(unknown)}")
        return self

    @model_validator(mode="after")
    def _index_must_live_outside_the_corpus(self) -> Settings:
        """Keep the index out of the corpus.

        An index written inside the corpus would be picked up by the next scan
        and, worse, would deposit generated files into the user's document
        folder.
        """
        if self.index_dir == self.corpus_dir or self.index_dir.is_relative_to(self.corpus_dir):
            raise ValueError(
                f"index_dir ({self.index_dir}) must not be inside corpus_dir "
                f"({self.corpus_dir})"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loading them on first use.

    Cached so that configuration is read and validated exactly once. Tests
    construct :class:`Settings` directly instead of going through this.
    """
    return Settings()  # type: ignore[call-arg]  # values come from the environment
