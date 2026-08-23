"""Shared test fixtures and builders.

Every fixture here is synthetic. No file, path or text from a real document
corpus appears anywhere in the test suite — see ``docs/decisions.md``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from local_rag.models import (
    Chunk,
    ChunkMetadata,
    DocumentMetadata,
    ExtractedDocument,
    PageText,
    SourceType,
)

FIXED_TIME = datetime(2024, 3, 5, 12, 30, tzinfo=timezone.utc)


def make_document_metadata(**overrides: Any) -> DocumentMetadata:
    """Build a :class:`DocumentMetadata` with sensible defaults.

    Keyword overrides replace individual fields, so a test states only the
    attribute it actually cares about.
    """
    defaults: dict[str, Any] = {
        "relative_path": "contracts/lease.pdf",
        "source_type": SourceType.DOCUMENT,
        "file_extension": ".pdf",
        "size_bytes": 2048,
        "modified_at": FIXED_TIME,
        "content_hash": "a" * 64,
        "page_count": 3,
    }
    return DocumentMetadata(**{**defaults, **overrides})


def make_chunk(text: str = "some text", **metadata_overrides: Any) -> Chunk:
    """Build a :class:`Chunk` wrapping ``text`` with default provenance."""
    defaults: dict[str, Any] = {
        "document": make_document_metadata(),
        "chunk_index": 0,
        "start_char": 0,
        "end_char": len(text),
    }
    return Chunk(page_content=text, metadata=ChunkMetadata(**{**defaults, **metadata_overrides}))


@pytest.fixture
def document_metadata() -> DocumentMetadata:
    """Provenance for a three-page synthetic PDF."""
    return make_document_metadata()


@pytest.fixture
def extracted_document(document_metadata: DocumentMetadata) -> ExtractedDocument:
    """A synthetic three-page document with distinguishable page text."""
    return ExtractedDocument(
        metadata=document_metadata,
        pages=(
            PageText(page_number=1, text="First page."),
            PageText(page_number=2, text="Second page."),
            PageText(page_number=3, text="Third page."),
        ),
    )
