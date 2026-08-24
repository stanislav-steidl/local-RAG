"""Core data structures shared by every stage of the pipeline.

These types are the contract between ingestion, chunking, embedding, storage and
retrieval. Keeping them free of any dependency on a parser, model or database is
what allows each stage to be replaced — or tested with a fake — in isolation.

Two deliberate choices are worth stating:

*Dataclasses, not Pydantic models.* Chunks are created in bulk on the indexing
hot path, where frozen slotted dataclasses are considerably cheaper than
validated models. Validation still happens, but only on the cheap invariants
that catch programming errors. Pydantic is reserved for
:mod:`local_rag.config`, where input genuinely comes from outside the program.

*Relative paths only.* A document is identified by its path relative to the
corpus root, never by an absolute path. The index therefore stays portable
across machines and cannot leak a home directory layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime

__all__ = [
    "Chunk",
    "ChunkMetadata",
    "DocumentMetadata",
    "ExtractedDocument",
    "PageText",
    "SourceType",
]


class SourceType(StrEnum):
    """The kind of artefact a piece of retrievable text originated from.

    Photos are not yet ingested, but the field exists from the outset so that
    adding them later is an additive change rather than a schema migration.
    """

    DOCUMENT = "document"
    PHOTO = "photo"


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """Provenance of a single source file in the corpus.

    Attributes:
        relative_path: Location of the file relative to the corpus root, using
            forward slashes so an index built on Windows reads correctly on
            POSIX systems.
        source_type: Whether the file is a document or a photo.
        file_extension: Lower-cased extension including the leading dot.
        size_bytes: Size of the file on disk.
        modified_at: Filesystem modification time, timezone-aware.
        content_hash: Digest of the file's bytes, used to detect which files
            actually changed so re-indexing can skip the rest.
        page_count: Number of pages, where the format has a concept of pages.
        extra: Format-specific metadata that does not warrant a dedicated
            field — EXIF and GPS data for photos, for example.
    """

    relative_path: str
    source_type: SourceType
    file_extension: str
    size_bytes: int
    modified_at: datetime
    content_hash: str
    page_count: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate invariants that only a programming error could violate."""
        if not self.relative_path:
            raise ValueError("relative_path must not be empty")
        if "\\" in self.relative_path:
            raise ValueError(f"relative_path must use forward slashes, got {self.relative_path!r}")
        if self.size_bytes < 0:
            raise ValueError(f"size_bytes must be non-negative, got {self.size_bytes}")
        if self.page_count is not None and self.page_count < 0:
            raise ValueError(f"page_count must be non-negative, got {self.page_count}")
        if self.modified_at.tzinfo is None:
            raise ValueError("modified_at must be timezone-aware")

    @property
    def file_name(self) -> str:
        """The final component of :attr:`relative_path`."""
        return self.relative_path.rsplit("/", maxsplit=1)[-1]


@dataclass(frozen=True, slots=True)
class PageText:
    """Text extracted from one page of a paginated document.

    Retaining page boundaries lets a retrieved chunk cite the page it came from,
    which is what makes an answer verifiable against the original file.

    Attributes:
        page_number: One-based page index, matching what a reader would see.
        text: Text extracted from that page.
    """

    page_number: int
    text: str

    def __post_init__(self) -> None:
        """Validate that pagination is one-based."""
        if self.page_number < 1:
            raise ValueError(f"page_number is one-based, got {self.page_number}")


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """The full text of one source file, as produced by a parser.

    Attributes:
        metadata: Provenance of the file the text came from.
        pages: Per-page text for paginated formats. Formats without pages
            (plain text, DOCX) report a single page.
    """

    metadata: DocumentMetadata
    pages: tuple[PageText, ...]

    @property
    def text(self) -> str:
        """All pages joined into one string, separated by blank lines."""
        return "\n\n".join(page.text for page in self.pages)

    @property
    def is_empty(self) -> bool:
        """Whether extraction yielded no usable text.

        A true result usually means a scanned document whose pages are images,
        and is the signal that the file needs OCR rather than text extraction.
        """
        return not self.text.strip()


@dataclass(frozen=True, slots=True)
class ChunkMetadata:
    """Where a chunk sits within its source document.

    Attributes:
        document: Provenance of the file the chunk was cut from.
        chunk_index: Zero-based position of this chunk within the document.
        start_char: Offset of the chunk's first character in the source text.
        end_char: Offset one past the chunk's last character.
        page_number: Page the chunk starts on, when the format has pages.
    """

    document: DocumentMetadata
    chunk_index: int
    start_char: int
    end_char: int
    page_number: int | None = None

    def __post_init__(self) -> None:
        """Validate that the character span is coherent."""
        if self.chunk_index < 0:
            raise ValueError(f"chunk_index must be non-negative, got {self.chunk_index}")
        if self.start_char < 0:
            raise ValueError(f"start_char must be non-negative, got {self.start_char}")
        if self.end_char < self.start_char:
            raise ValueError(
                f"end_char ({self.end_char}) must not precede start_char ({self.start_char})"
            )


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit of text together with its provenance.

    The ``page_content`` / ``metadata`` naming is not incidental: it matches
    ``langchain_core.documents.Document``, so adopting LangChain later means
    writing an adapter rather than reshaping the pipeline. See
    ``docs/decisions.md`` for the full reasoning.

    Attributes:
        page_content: The chunk's text, as embedded and as shown to the user.
        metadata: Provenance and position of the chunk.
    """

    page_content: str
    metadata: ChunkMetadata

    @property
    def citation(self) -> str:
        """A short human-readable reference to where this text came from.

        Renders as ``contracts/lease.pdf (p. 3)`` for paginated sources and
        ``notes.txt`` otherwise.
        """
        source = self.metadata.document.relative_path
        page = self.metadata.page_number
        return f"{source} (p. {page})" if page is not None else source
