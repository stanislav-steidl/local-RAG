"""Tests for the composition of scanning and parsing into loaded documents."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

import pytest

from local_rag.ingest.base import DocumentParseError, DocumentParser, UnsupportedFormatError
from local_rag.ingest.loader import load_corpus, load_document
from local_rag.ingest.registry import ParserRegistry
from local_rag.models import PageText

from .synthetic import make_docx, make_pdf


class PaginatedParser(DocumentParser):
    """Reports a fixed three pages, standing in for a real paginated format."""

    extensions: ClassVar[frozenset[str]] = frozenset({".paged"})
    is_paginated: ClassVar[bool] = True

    def parse(self, path: Path) -> tuple[PageText, ...]:
        return tuple(PageText(page_number=n, text=f"page {n}") for n in range(1, 4))


class FlatParser(DocumentParser):
    """Reports a single page, standing in for an unpaginated format."""

    extensions: ClassVar[frozenset[str]] = frozenset({".flat"})
    is_paginated: ClassVar[bool] = False

    def parse(self, path: Path) -> tuple[PageText, ...]:
        return (PageText(page_number=1, text="flat text"),)


class ExplodingParser(DocumentParser):
    """Always fails, standing in for a corrupt file."""

    extensions: ClassVar[frozenset[str]] = frozenset({".boom"})
    is_paginated: ClassVar[bool] = False

    def parse(self, path: Path) -> tuple[PageText, ...]:
        raise DocumentParseError(f"unreadable: {path.name}")


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """An empty corpus root."""
    root = tmp_path / "corpus"
    root.mkdir()
    return root


@pytest.fixture
def registry() -> ParserRegistry:
    """A registry of fakes, so loader tests do not depend on real file formats."""
    return ParserRegistry([PaginatedParser(), FlatParser(), ExplodingParser()])


def touch(root: Path, relative: str, content: str = "x") -> Path:
    """Create a file under ``root``."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestLoadDocument:
    def test_combines_metadata_with_extracted_pages(
        self, corpus: Path, registry: ParserRegistry
    ) -> None:
        path = touch(corpus, "contracts/a.paged")

        document = load_document(path, corpus, registry)

        assert document.metadata.relative_path == "contracts/a.paged"
        assert len(document.pages) == 3
        assert document.text.startswith("page 1")

    def test_page_count_is_recorded_for_paginated_formats(
        self, corpus: Path, registry: ParserRegistry
    ) -> None:
        path = touch(corpus, "a.paged")

        assert load_document(path, corpus, registry).metadata.page_count == 3

    def test_page_count_stays_unset_for_unpaginated_formats(
        self, corpus: Path, registry: ParserRegistry
    ) -> None:
        """Otherwise every DOCX citation would claim a meaningless 'page 1'."""
        path = touch(corpus, "a.flat")

        assert load_document(path, corpus, registry).metadata.page_count is None

    def test_unsupported_extension_raises(self, corpus: Path, registry: ParserRegistry) -> None:
        path = touch(corpus, "a.zip")

        with pytest.raises(UnsupportedFormatError):
            load_document(path, corpus, registry)

    def test_parse_failure_propagates(self, corpus: Path, registry: ParserRegistry) -> None:
        """A caller loading one named file deserves the error, not a silent skip."""
        path = touch(corpus, "a.boom")

        with pytest.raises(DocumentParseError, match="unreadable"):
            load_document(path, corpus, registry)

    def test_defaults_to_the_real_registry(self, corpus: Path) -> None:
        make_pdf(corpus / "real.pdf", ["Hello"])

        document = load_document(corpus / "real.pdf", corpus, None)

        assert document.metadata.page_count == 1
        assert "Hello" in document.text


class TestLoadCorpus:
    def test_loads_every_supported_document(self, corpus: Path, registry: ParserRegistry) -> None:
        touch(corpus, "a.paged")
        touch(corpus, "nested/b.flat")

        loaded = list(load_corpus(corpus, registry))

        assert [doc.metadata.relative_path for doc in loaded] == ["a.paged", "nested/b.flat"]

    def test_unsupported_files_are_never_opened(
        self, corpus: Path, registry: ParserRegistry
    ) -> None:
        touch(corpus, "a.paged")
        touch(corpus, "archive.zip")
        touch(corpus, "image.jpg")

        loaded = list(load_corpus(corpus, registry))

        assert [doc.metadata.relative_path for doc in loaded] == ["a.paged"]

    def test_a_corrupt_document_is_skipped_not_fatal(
        self, corpus: Path, registry: ParserRegistry, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One bad file must not abort an indexing run over gigabytes."""
        touch(corpus, "good.paged")
        touch(corpus, "bad.boom")
        touch(corpus, "also_good.flat")

        with caplog.at_level(logging.WARNING):
            loaded = list(load_corpus(corpus, registry))

        assert sorted(doc.metadata.relative_path for doc in loaded) == [
            "also_good.flat",
            "good.paged",
        ]
        assert "bad.boom" in caplog.text

    def test_empty_corpus_yields_nothing(self, corpus: Path, registry: ParserRegistry) -> None:
        assert list(load_corpus(corpus, registry)) == []

    def test_missing_root_raises_rather_than_yielding_nothing(
        self, tmp_path: Path, registry: ParserRegistry
    ) -> None:
        """A mistyped corpus path must fail, not silently index zero documents.

        os.walk over a missing directory yields nothing at all, so without an
        explicit check the caller cannot tell an empty corpus from a wrong path.
        """
        with pytest.raises(FileNotFoundError, match="does not exist"):
            list(load_corpus(tmp_path / "absent", registry))

    def test_root_that_is_a_file_raises(self, corpus: Path, registry: ParserRegistry) -> None:
        path = touch(corpus, "a.paged")

        with pytest.raises(NotADirectoryError, match="not a directory"):
            list(load_corpus(path, registry))

    def test_is_lazy(self, corpus: Path, registry: ParserRegistry) -> None:
        """A multi-gigabyte corpus must never be materialised in memory."""
        touch(corpus, "a.paged")

        assert not isinstance(load_corpus(corpus, registry), list)

    def test_registry_filter_mismatch_degrades_to_a_skip(
        self, corpus: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A registry advertising more than it resolves must not abort the run."""

        class LyingRegistry(ParserRegistry):
            @property
            def supported_extensions(self) -> frozenset[str]:
                return frozenset({".paged", ".unclaimed"})

        touch(corpus, "a.paged")
        touch(corpus, "b.unclaimed")

        with caplog.at_level(logging.DEBUG):
            loaded = list(load_corpus(corpus, LyingRegistry([PaginatedParser()])))

        assert [doc.metadata.relative_path for doc in loaded] == ["a.paged"]
        assert "b.unclaimed" in caplog.text

    def test_end_to_end_over_real_formats(self, corpus: Path) -> None:
        """The default registry against genuinely encoded files, not fakes."""
        make_pdf(corpus / "contract.pdf", ["Page one text", "Page two text"])
        make_docx(corpus / "notes.docx", paragraphs=["Some note"])
        (corpus / "plain.txt").write_text("plain content", encoding="utf-8")

        loaded = {doc.metadata.relative_path: doc for doc in load_corpus(corpus)}

        assert set(loaded) == {"contract.pdf", "notes.docx", "plain.txt"}
        assert loaded["contract.pdf"].metadata.page_count == 2
        assert loaded["notes.docx"].metadata.page_count is None
        assert "Some note" in loaded["notes.docx"].text
        assert loaded["plain.txt"].text == "plain content"
        assert not any(doc.is_empty for doc in loaded.values())
