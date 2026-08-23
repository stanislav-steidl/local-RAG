"""The parser abstraction and the errors ingestion can raise.

A parser's single responsibility is turning one file into pages of text. It
does not decide whether the file should be indexed, does not build metadata,
and does not care where the file came from — which is what allows a test to
substitute a fake parser without touching the filesystem.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pathlib import Path

    from local_rag.models import PageText

__all__ = [
    "DocumentParseError",
    "DocumentParser",
    "IngestionError",
    "MissingDependencyError",
    "UnsupportedFormatError",
]


class IngestionError(Exception):
    """Base class for every failure raised while ingesting a document."""


class UnsupportedFormatError(IngestionError):
    """Raised when no registered parser handles a file's extension."""


class DocumentParseError(IngestionError):
    """Raised when a file's format is supported but its contents cannot be read.

    Distinct from :class:`UnsupportedFormatError` because the two call for
    different responses: an unsupported format is expected and uninteresting,
    whereas a corrupt file the pipeline claims to support is worth reporting.
    """


class MissingDependencyError(IngestionError):
    """Raised when a parser's optional dependency is not installed.

    Parsing libraries live behind the ``parsing`` extra, so importing
    :mod:`local_rag.ingest` must not require them. The failure is therefore
    deferred to the moment a parser is actually used, where it can name the
    extra that would fix it.
    """


class DocumentParser(ABC):
    """Extracts text from one family of file formats.

    Attributes:
        extensions: Lower-cased extensions this parser handles, each including
            the leading dot.
        is_paginated: Whether the format has a genuine concept of pages. False
            for formats such as DOCX and plain text, whose page breaks are a
            rendering decision rather than a property of the file. Callers use
            this to decide whether a page number is meaningful enough to cite.
    """

    extensions: ClassVar[frozenset[str]]
    is_paginated: ClassVar[bool]

    @abstractmethod
    def parse(self, path: Path) -> tuple[PageText, ...]:
        r"""Extract text from ``path``.

        An empty result is a legitimate outcome, not an error: a scanned
        document has pages that contain no extractable text, and recognising
        that is how such files get routed to OCR later.

        Args:
            path: File to read.

        Returns:
            One :class:`~local_rag.models.PageText` per page, in reading order.
            Unpaginated formats return a single entry numbered 1. Text must use
            ``\n`` line endings; every implementation normalises CRLF and lone
            CR so that later stages need only recognise one spelling of a line
            break.

        Raises:
            DocumentParseError: If the file cannot be parsed.
            MissingDependencyError: If the parser's optional dependency is
                not installed.
        """

    def __repr__(self) -> str:
        """Name the parser and the extensions it claims."""
        claimed = ", ".join(sorted(self.extensions))
        return f"{type(self).__name__}({claimed})"
