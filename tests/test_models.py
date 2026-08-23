"""Tests for the core pipeline data structures."""

from __future__ import annotations

import dataclasses
from datetime import datetime

import pytest

from local_rag.models import (
    ChunkMetadata,
    DocumentMetadata,
    ExtractedDocument,
    PageText,
    SourceType,
)

from .conftest import FIXED_TIME, make_chunk, make_document_metadata


class TestDocumentMetadata:
    def test_file_name_is_derived_from_relative_path(self) -> None:
        meta = make_document_metadata(relative_path="contracts/2022/lease.pdf")
        assert meta.file_name == "lease.pdf"

    def test_file_name_of_a_root_level_file(self) -> None:
        meta = make_document_metadata(relative_path="lease.pdf")
        assert meta.file_name == "lease.pdf"

    def test_is_immutable(self) -> None:
        meta = make_document_metadata()
        with pytest.raises(dataclasses.FrozenInstanceError):
            meta.relative_path = "other.pdf"  # type: ignore[misc]

    def test_rejects_empty_relative_path(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            make_document_metadata(relative_path="")

    def test_rejects_backslash_paths(self) -> None:
        """Windows-style separators would make an index unreadable on POSIX."""
        with pytest.raises(ValueError, match="forward slashes"):
            make_document_metadata(relative_path="contracts\\lease.pdf")

    def test_rejects_negative_size(self) -> None:
        with pytest.raises(ValueError, match="size_bytes"):
            make_document_metadata(size_bytes=-1)

    def test_rejects_negative_page_count(self) -> None:
        with pytest.raises(ValueError, match="page_count"):
            make_document_metadata(page_count=-1)

    def test_allows_unknown_page_count(self) -> None:
        """Formats without pagination legitimately report no page count."""
        assert make_document_metadata(page_count=None).page_count is None

    def test_rejects_naive_timestamps(self) -> None:
        """A naive timestamp silently misorders documents across DST changes."""
        naive = datetime(2024, 3, 5, 12, 30)
        with pytest.raises(ValueError, match="timezone-aware"):
            make_document_metadata(modified_at=naive)

    def test_accepts_zero_byte_file(self) -> None:
        assert make_document_metadata(size_bytes=0).size_bytes == 0

    def test_extra_defaults_to_empty_and_is_not_shared(self) -> None:
        """Each instance must own its ``extra`` mapping, not share a default."""
        first = make_document_metadata()
        second = make_document_metadata()
        assert first.extra == {}
        first.extra["gps"] = (50.08, 14.44)
        assert second.extra == {}

    def test_source_type_serialises_as_its_string_value(self) -> None:
        """Storage layers persist the value, so it must round-trip as text."""
        assert SourceType.DOCUMENT.value == "document"
        assert SourceType("photo") is SourceType.PHOTO


class TestPageText:
    def test_rejects_zero_page_number(self) -> None:
        with pytest.raises(ValueError, match="one-based"):
            PageText(page_number=0, text="text")

    def test_accepts_empty_text(self) -> None:
        """A blank page is legitimate; it is not an extraction failure."""
        assert PageText(page_number=1, text="").text == ""


class TestExtractedDocument:
    def test_text_joins_pages_with_blank_lines(self, extracted_document: ExtractedDocument) -> None:
        assert extracted_document.text == "First page.\n\nSecond page.\n\nThird page."

    def test_text_of_a_document_with_no_pages(self, document_metadata: DocumentMetadata) -> None:
        assert ExtractedDocument(metadata=document_metadata, pages=()).text == ""

    def test_is_empty_is_false_when_text_was_extracted(
        self, extracted_document: ExtractedDocument
    ) -> None:
        assert not extracted_document.is_empty

    def test_is_empty_is_true_without_pages(self, document_metadata: DocumentMetadata) -> None:
        assert ExtractedDocument(metadata=document_metadata, pages=()).is_empty

    def test_is_empty_is_true_for_whitespace_only_pages(
        self, document_metadata: DocumentMetadata
    ) -> None:
        """A scanned PDF yields whitespace, and must be routed to OCR."""
        scanned = ExtractedDocument(
            metadata=document_metadata,
            pages=(PageText(page_number=1, text="  \n\t "),),
        )
        assert scanned.is_empty


class TestChunkMetadata:
    def test_rejects_negative_chunk_index(self) -> None:
        with pytest.raises(ValueError, match="chunk_index"):
            ChunkMetadata(
                document=make_document_metadata(), chunk_index=-1, start_char=0, end_char=1
            )

    def test_rejects_negative_start_char(self) -> None:
        with pytest.raises(ValueError, match="start_char"):
            ChunkMetadata(
                document=make_document_metadata(), chunk_index=0, start_char=-1, end_char=1
            )

    def test_rejects_end_before_start(self) -> None:
        with pytest.raises(ValueError, match="must not precede"):
            ChunkMetadata(
                document=make_document_metadata(), chunk_index=0, start_char=10, end_char=5
            )

    def test_allows_empty_span(self) -> None:
        """A zero-length span is degenerate but not an error."""
        meta = ChunkMetadata(
            document=make_document_metadata(), chunk_index=0, start_char=7, end_char=7
        )
        assert meta.start_char == meta.end_char


class TestChunk:
    def test_citation_includes_page_when_known(self) -> None:
        chunk = make_chunk(page_number=3)
        assert chunk.citation == "contracts/lease.pdf (p. 3)"

    def test_citation_omits_page_when_unknown(self) -> None:
        chunk = make_chunk(page_number=None)
        assert chunk.citation == "contracts/lease.pdf"

    def test_exposes_langchain_document_shape(self) -> None:
        """``page_content`` + ``metadata`` keeps a future LangChain adapter trivial."""
        chunk = make_chunk("hello")
        assert chunk.page_content == "hello"
        assert isinstance(chunk.metadata, ChunkMetadata)

    def test_is_immutable(self) -> None:
        chunk = make_chunk()
        with pytest.raises(dataclasses.FrozenInstanceError):
            chunk.page_content = "changed"  # type: ignore[misc]

    def test_retains_provenance_through_the_pipeline(self) -> None:
        """Provenance must survive chunking so answers can cite their source."""
        chunk = make_chunk("text", chunk_index=4, start_char=100, end_char=104)
        assert chunk.metadata.document.modified_at == FIXED_TIME
        assert chunk.metadata.document.content_hash == "a" * 64
        assert chunk.metadata.chunk_index == 4

    def test_equality_is_structural(self) -> None:
        """Value semantics let tests compare chunks directly."""
        assert make_chunk("same") == make_chunk("same")
        assert make_chunk("a") != make_chunk("b")
