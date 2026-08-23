"""Mapping from file extension to the parser that handles it."""

from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag.ingest.base import UnsupportedFormatError
from local_rag.ingest.parsers import DocxParser, PdfParser, TextParser

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from local_rag.ingest.base import DocumentParser

__all__ = ["ParserRegistry", "default_registry"]


class ParserRegistry:
    """Resolves a file to the parser responsible for it.

    Registering two parsers for the same extension is rejected rather than
    silently resolved by ordering, since which one wins would otherwise depend
    on construction order and be invisible at the call site.
    """

    def __init__(self, parsers: Iterable[DocumentParser]) -> None:
        """Build a registry from parsers.

        Args:
            parsers: Parsers to register.

        Raises:
            ValueError: If two parsers claim the same extension.
        """
        self._by_extension: dict[str, DocumentParser] = {}
        for parser in parsers:
            for extension in parser.extensions:
                key = extension.lower()
                existing = self._by_extension.get(key)
                if existing is not None:
                    raise ValueError(f"{key} is claimed by both {existing!r} and {parser!r}")
                self._by_extension[key] = parser

    @property
    def supported_extensions(self) -> frozenset[str]:
        """Every extension this registry can parse, lower-cased."""
        return frozenset(self._by_extension)

    def for_extension(self, extension: str) -> DocumentParser:
        """Return the parser registered for ``extension``.

        Args:
            extension: Extension including the leading dot, any case.

        Returns:
            The parser responsible for that extension.

        Raises:
            UnsupportedFormatError: If no parser is registered for it.
        """
        parser = self._by_extension.get(extension.lower())
        if parser is None:
            supported = ", ".join(sorted(self._by_extension)) or "none"
            raise UnsupportedFormatError(
                f"no parser for {extension!r}; supported extensions: {supported}"
            )
        return parser

    def for_path(self, path: Path) -> DocumentParser:
        """Return the parser for ``path`` based on its extension.

        Args:
            path: File to resolve a parser for.

        Returns:
            The parser responsible for the file.

        Raises:
            UnsupportedFormatError: If no parser handles the file's extension.
        """
        return self.for_extension(path.suffix)

    def __contains__(self, extension: object) -> bool:
        """Whether ``extension`` has a registered parser."""
        return isinstance(extension, str) and extension.lower() in self._by_extension

    def __len__(self) -> int:
        """Number of extensions the registry resolves."""
        return len(self._by_extension)

    def __repr__(self) -> str:
        """Show the extensions covered."""
        return f"{type(self).__name__}({', '.join(sorted(self._by_extension))})"


def default_registry() -> ParserRegistry:
    """Return a registry covering every format the pipeline supports today.

    OCR-backed parsers for scanned PDFs and photographed documents will be
    added here once that increment lands.
    """
    return ParserRegistry([TextParser(), PdfParser(), DocxParser()])
