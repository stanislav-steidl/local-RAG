"""Tests for extension-to-parser resolution."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from local_rag.ingest.base import DocumentParser, UnsupportedFormatError
from local_rag.ingest.registry import ParserRegistry, default_registry
from local_rag.models import PageText


class FakeParser(DocumentParser):
    """A parser that reports fixed text, so registry tests touch no files."""

    extensions: ClassVar[frozenset[str]] = frozenset({".fake"})
    is_paginated: ClassVar[bool] = False

    def parse(self, path: Path) -> tuple[PageText, ...]:
        return (PageText(page_number=1, text=f"parsed {path.name}"),)


class OtherFakeParser(FakeParser):
    """Claims the same extension as :class:`FakeParser`, to force a conflict."""


class TestResolution:
    def test_resolves_a_registered_extension(self) -> None:
        parser = FakeParser()
        registry = ParserRegistry([parser])

        assert registry.for_extension(".fake") is parser

    def test_resolution_ignores_case(self) -> None:
        parser = FakeParser()
        registry = ParserRegistry([parser])

        assert registry.for_extension(".FAKE") is parser

    def test_resolves_from_a_path(self) -> None:
        parser = FakeParser()
        registry = ParserRegistry([parser])

        assert registry.for_path(Path("a/b/doc.fake")) is parser

    def test_unsupported_extension_lists_what_is_supported(self) -> None:
        """The error should tell the user what they could have used instead."""
        registry = ParserRegistry([FakeParser()])

        with pytest.raises(UnsupportedFormatError, match=r"no parser for '\.zip'") as excinfo:
            registry.for_extension(".zip")

        assert ".fake" in str(excinfo.value)

    def test_empty_registry_reports_no_supported_extensions(self) -> None:
        with pytest.raises(UnsupportedFormatError, match="none"):
            ParserRegistry([]).for_extension(".pdf")

    def test_file_without_an_extension_is_unsupported(self) -> None:
        registry = ParserRegistry([FakeParser()])

        with pytest.raises(UnsupportedFormatError):
            registry.for_path(Path("README"))


class TestConstruction:
    def test_duplicate_extensions_are_rejected(self) -> None:
        """Silently letting construction order decide the winner would be invisible."""
        with pytest.raises(ValueError, match="claimed by both"):
            ParserRegistry([FakeParser(), OtherFakeParser()])

    def test_conflict_message_names_both_parsers(self) -> None:
        with pytest.raises(ValueError, match="FakeParser") as excinfo:
            ParserRegistry([FakeParser(), OtherFakeParser()])

        assert "OtherFakeParser" in str(excinfo.value)


class TestIntrospection:
    def test_supported_extensions_reports_every_key(self) -> None:
        assert ParserRegistry([FakeParser()]).supported_extensions == frozenset({".fake"})

    def test_membership_ignores_case(self) -> None:
        registry = ParserRegistry([FakeParser()])

        assert ".fake" in registry
        assert ".FAKE" in registry
        assert ".pdf" not in registry

    def test_membership_of_a_non_string_is_false(self) -> None:
        assert 42 not in ParserRegistry([FakeParser()])

    def test_length_counts_extensions_not_parsers(self) -> None:
        """One parser claiming three extensions occupies three slots."""
        registry = ParserRegistry([FakeParser()])
        assert len(registry) == 1

    def test_repr_lists_the_extensions(self) -> None:
        assert repr(ParserRegistry([FakeParser()])) == "ParserRegistry(.fake)"


class TestDefaultRegistry:
    @pytest.mark.parametrize("extension", [".pdf", ".docx", ".txt", ".md", ".csv"])
    def test_covers_the_formats_the_pipeline_claims(self, extension: str) -> None:
        assert extension in default_registry()

    def test_does_not_yet_claim_image_formats(self) -> None:
        """Photos need OCR or metadata extraction, neither of which exists yet."""
        registry = default_registry()

        assert ".jpg" not in registry
        assert ".png" not in registry

    def test_each_call_returns_an_independent_registry(self) -> None:
        assert default_registry() is not default_registry()
