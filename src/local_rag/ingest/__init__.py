"""Discovery and text extraction for the document corpus.

The stage is split so each part can be tested without the others: the scanner
decides *which* files exist and what the filesystem knows about them, parsers
turn one file into text, and the loader composes the two.

Importing this package does not require the ``parsing`` extra. Parser
dependencies are imported on first use, so a missing library surfaces as a
:class:`MissingDependencyError` naming the extra that would fix it.
"""

from __future__ import annotations

from local_rag.ingest.base import (
    DocumentParseError,
    DocumentParser,
    IngestionError,
    MissingDependencyError,
    UnsupportedFormatError,
)
from local_rag.ingest.loader import load_corpus, load_document
from local_rag.ingest.parsers import DocxParser, PdfParser, TextParser
from local_rag.ingest.registry import ParserRegistry, default_registry
from local_rag.ingest.scanner import (
    IGNORED_FILE_NAMES,
    IGNORED_NAME_PREFIXES,
    IMAGE_EXTENSIONS,
    build_document_metadata,
    classify_source_type,
    compute_content_hash,
    iter_source_files,
    scan_corpus,
    validate_corpus_root,
)

__all__ = [
    "IGNORED_FILE_NAMES",
    "IGNORED_NAME_PREFIXES",
    "IMAGE_EXTENSIONS",
    "DocumentParseError",
    "DocumentParser",
    "DocxParser",
    "IngestionError",
    "MissingDependencyError",
    "ParserRegistry",
    "PdfParser",
    "TextParser",
    "UnsupportedFormatError",
    "build_document_metadata",
    "classify_source_type",
    "compute_content_hash",
    "default_registry",
    "iter_source_files",
    "load_corpus",
    "load_document",
    "scan_corpus",
    "validate_corpus_root",
]
