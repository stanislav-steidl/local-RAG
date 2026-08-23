"""Discovery and text extraction for the document corpus.

The stage is split in two so each half can be tested without the other: the
scanner decides *which* files exist and what is known about them from the
filesystem alone, while parsers turn one file into text.
"""

from __future__ import annotations

from local_rag.ingest.scanner import (
    IGNORED_FILE_NAMES,
    IGNORED_NAME_PREFIXES,
    IMAGE_EXTENSIONS,
    build_document_metadata,
    compute_content_hash,
    iter_source_files,
    scan_corpus,
)

__all__ = [
    "IGNORED_FILE_NAMES",
    "IGNORED_NAME_PREFIXES",
    "IMAGE_EXTENSIONS",
    "build_document_metadata",
    "compute_content_hash",
    "iter_source_files",
    "scan_corpus",
]
