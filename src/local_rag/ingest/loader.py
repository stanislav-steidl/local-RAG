"""Composition of scanning and parsing into loaded documents.

This is the seam where the two halves of ingestion meet: the scanner says what
exists, a parser says what it contains, and the loader combines them into the
:class:`~local_rag.models.ExtractedDocument` that the chunker consumes.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

from local_rag.ingest.base import IngestionError, UnsupportedFormatError
from local_rag.ingest.registry import default_registry
from local_rag.ingest.scanner import build_document_metadata, iter_source_files
from local_rag.models import ExtractedDocument

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from local_rag.ingest.registry import ParserRegistry

__all__ = ["load_corpus", "load_document"]

logger = logging.getLogger(__name__)


def load_document(
    path: Path,
    root: Path,
    registry: ParserRegistry | None = None,
) -> ExtractedDocument:
    """Read one file into an :class:`ExtractedDocument`.

    ``page_count`` is filled in only for genuinely paginated formats. For DOCX
    and plain text it stays ``None``, which is what later tells the chunker not
    to attach a misleading "page 1" to every citation.

    Args:
        path: File to load.
        root: Corpus root, used to derive the stored relative path.
        registry: Parsers to resolve against. Defaults to
            :func:`~local_rag.ingest.registry.default_registry`.

    Returns:
        The file's metadata together with its extracted pages.

    Raises:
        UnsupportedFormatError: If no parser handles the file's extension.
        DocumentParseError: If the file cannot be parsed.
        MissingDependencyError: If the parser's optional dependency is missing.
        OSError: If the file cannot be inspected.
        ValueError: If ``path`` is not beneath ``root``.
    """
    resolved = registry if registry is not None else default_registry()
    parser = resolved.for_path(path)
    metadata = build_document_metadata(path, root)
    pages = parser.parse(path)

    if parser.is_paginated:
        metadata = dataclasses.replace(metadata, page_count=len(pages))

    return ExtractedDocument(metadata=metadata, pages=pages)


def load_corpus(
    root: Path,
    registry: ParserRegistry | None = None,
) -> Iterator[ExtractedDocument]:
    """Load every supported document beneath ``root``.

    Only extensions the registry recognises are visited, so unsupported files
    are never opened. A file that *is* supported but fails to parse is logged
    and skipped: one corrupt PDF must not abort an indexing run.

    Args:
        root: Corpus root to load.
        registry: Parsers to resolve against. Defaults to
            :func:`~local_rag.ingest.registry.default_registry`.

    Yields:
        One loaded document per readable, parseable file, in scan order.
    """
    resolved = registry if registry is not None else default_registry()

    for path in iter_source_files(root, extensions=resolved.supported_extensions):
        try:
            yield load_document(path, root, resolved)
        except UnsupportedFormatError:
            # Cannot normally happen: the walk is already filtered to supported
            # extensions. Guarded so a registry/filter mismatch degrades to a
            # skip rather than aborting the run.
            logger.debug("No parser for %s despite extension filter", path)
        except (IngestionError, OSError) as error:
            logger.warning("Skipping %s: %s", path, error)
