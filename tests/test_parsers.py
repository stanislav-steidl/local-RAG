"""Tests for the concrete document parsers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from local_rag.ingest.base import DocumentParseError, MissingDependencyError
from local_rag.ingest.parsers import DocxParser, PdfParser, TextParser

from .synthetic import CZECH_SAMPLE, make_docx, make_pdf, make_scanned_pdf


class TestTextParser:
    def test_declares_its_capabilities(self) -> None:
        parser = TextParser()
        assert ".txt" in parser.extensions
        assert parser.is_paginated is False

    def test_reads_utf8(self, tmp_path: Path) -> None:
        path = tmp_path / "note.txt"
        path.write_text(CZECH_SAMPLE, encoding="utf-8")

        pages = TextParser().parse(path)

        assert len(pages) == 1
        assert pages[0].text == CZECH_SAMPLE
        assert pages[0].page_number == 1

    def test_strips_the_utf8_byte_order_mark(self, tmp_path: Path) -> None:
        """Notepad writes a BOM; it must not survive into the indexed text."""
        path = tmp_path / "note.txt"
        path.write_bytes(b"\xef\xbb\xbf" + CZECH_SAMPLE.encode("utf-8"))

        assert TextParser().parse(path)[0].text == CZECH_SAMPLE

    def test_reads_legacy_windows_central_european_text(self, tmp_path: Path) -> None:
        """Older Czech documents are cp1250; decoding them as UTF-8 fails outright."""
        path = tmp_path / "stary.txt"
        path.write_bytes(CZECH_SAMPLE.encode("cp1250"))

        assert TextParser().parse(path)[0].text == CZECH_SAMPLE

    def test_crlf_line_endings_are_normalised(self, tmp_path: Path) -> None:
        r"""Chunking looks for "\n\n" paragraph breaks, which CRLF does not contain.

        Without normalisation a Windows text file would lose every paragraph
        boundary to a mere line boundary during chunking.
        """
        path = tmp_path / "windows.txt"
        path.write_bytes(b"First paragraph.\r\n\r\nSecond paragraph.\r\n")

        text = TextParser().parse(path)[0].text

        assert "\r" not in text
        assert text == "First paragraph.\n\nSecond paragraph.\n"

    def test_lone_carriage_returns_are_normalised(self, tmp_path: Path) -> None:
        path = tmp_path / "classic.txt"
        path.write_bytes(b"First line.\rSecond line.")

        assert TextParser().parse(path)[0].text == "First line.\nSecond line."

    def test_empty_file_yields_no_pages(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.txt"
        path.write_text("", encoding="utf-8")

        assert TextParser().parse(path) == ()

    def test_missing_file_raises_a_parse_error(self, tmp_path: Path) -> None:
        """Filesystem failures are reported in the pipeline's own vocabulary."""
        with pytest.raises(DocumentParseError, match="cannot read"):
            TextParser().parse(tmp_path / "absent.txt")


class TestPdfParser:
    def test_declares_its_capabilities(self) -> None:
        parser = PdfParser()
        assert parser.extensions == frozenset({".pdf"})
        assert parser.is_paginated is True

    def test_extracts_one_entry_per_page(self, tmp_path: Path) -> None:
        path = make_pdf(tmp_path / "doc.pdf", ["First page", "Second page", "Third page"])

        pages = PdfParser().parse(path)

        assert [page.page_number for page in pages] == [1, 2, 3]
        assert "First page" in pages[0].text
        assert "Third page" in pages[2].text

    def test_page_numbering_is_one_based(self, tmp_path: Path) -> None:
        path = make_pdf(tmp_path / "doc.pdf", ["only"])

        assert PdfParser().parse(path)[0].page_number == 1

    def test_scanned_pages_are_kept_as_empty_pages(self, tmp_path: Path) -> None:
        """Dropping them would desynchronise page numbers from the real document."""
        path = make_scanned_pdf(tmp_path / "scan.pdf", page_count=3)

        pages = PdfParser().parse(path)

        assert len(pages) == 3
        assert all(not page.text.strip() for page in pages)
        assert [page.page_number for page in pages] == [1, 2, 3]

    def test_corrupt_file_raises_a_parse_error(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.pdf"
        path.write_bytes(b"%PDF-1.4 this is not actually a pdf")

        with pytest.raises(DocumentParseError, match="cannot parse PDF"):
            PdfParser().parse(path)

    def test_missing_file_raises_a_parse_error(self, tmp_path: Path) -> None:
        with pytest.raises(DocumentParseError, match="cannot parse PDF"):
            PdfParser().parse(tmp_path / "absent.pdf")

    def test_missing_dependency_names_the_extra(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The error must tell the user how to fix it, not just that it broke."""
        monkeypatch.setitem(sys.modules, "pdfplumber", None)
        path = make_pdf(tmp_path / "doc.pdf", ["text"])

        with pytest.raises(MissingDependencyError, match=r"local-rag\[parsing\]"):
            PdfParser().parse(path)


class TestDocxParser:
    def test_declares_its_capabilities(self) -> None:
        parser = DocxParser()
        assert parser.extensions == frozenset({".docx"})
        assert parser.is_paginated is False

    def test_extracts_paragraphs_as_a_single_page(self, tmp_path: Path) -> None:
        """DOCX page breaks are a rendering decision, not a property of the file."""
        path = make_docx(tmp_path / "doc.docx", paragraphs=["First para", "Second para"])

        pages = DocxParser().parse(path)

        assert len(pages) == 1
        assert "First para" in pages[0].text
        assert "Second para" in pages[0].text

    def test_paragraphs_are_separated_by_a_blank_line(self, tmp_path: Path) -> None:
        """A single newline would be indistinguishable from a soft line break.

        The chunker prefers paragraph breaks over line breaks; joining blocks
        with one newline meant that preference could never apply to DOCX.
        """
        path = make_docx(tmp_path / "doc.docx", paragraphs=["First para", "Second para"])

        assert DocxParser().parse(path)[0].text == "First para\n\nSecond para"

    def test_table_rows_keep_their_cells_on_one_line(self, tmp_path: Path) -> None:
        """A label and its value must stay adjacent to be retrievable together."""
        path = make_docx(
            tmp_path / "invoice.docx",
            table=[["Polozka", "Castka"], ["Sluzba", "12 345 Kc"]],
        )

        assert DocxParser().parse(path)[0].text == "Polozka\tCastka\nSluzba\t12 345 Kc"

    def test_extracts_table_cells(self, tmp_path: Path) -> None:
        """Invoices keep the retrievable facts — amounts, dates — inside tables."""
        path = make_docx(
            tmp_path / "invoice.docx",
            paragraphs=["Faktura"],
            table=[["Polozka", "Castka"], ["Sluzba", "12 345 Kc"]],
        )

        text = DocxParser().parse(path)[0].text

        assert "12 345 Kc" in text
        assert "Polozka" in text

    def test_preserves_czech_diacritics(self, tmp_path: Path) -> None:
        path = make_docx(tmp_path / "doc.docx", paragraphs=[CZECH_SAMPLE])

        assert CZECH_SAMPLE in DocxParser().parse(path)[0].text

    def test_line_endings_are_normalised(self, tmp_path: Path) -> None:
        """Word stores soft line breaks as CR; every parser must emit LF only."""
        path = make_docx(tmp_path / "doc.docx", paragraphs=["First\rSecond"])

        assert "\r" not in DocxParser().parse(path)[0].text

    def test_document_without_text_yields_no_pages(self, tmp_path: Path) -> None:
        path = make_docx(tmp_path / "empty.docx", paragraphs=[])

        assert DocxParser().parse(path) == ()

    def test_whitespace_only_paragraphs_are_dropped(self, tmp_path: Path) -> None:
        path = make_docx(tmp_path / "blank.docx", paragraphs=["   ", "\t"])

        assert DocxParser().parse(path) == ()

    def test_corrupt_file_raises_a_parse_error(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.docx"
        path.write_bytes(b"not a zip archive at all")

        with pytest.raises(DocumentParseError, match="cannot parse DOCX"):
            DocxParser().parse(path)

    def test_missing_dependency_names_the_extra(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "docx", None)
        path = make_docx(tmp_path / "doc.docx", paragraphs=["text"])

        with pytest.raises(MissingDependencyError, match=r"local-rag\[parsing\]"):
            DocxParser().parse(path)


class TestParserRepr:
    def test_names_the_class_and_its_extensions(self) -> None:
        """Registry conflict errors embed this, so it must identify the parser."""
        assert repr(PdfParser()) == "PdfParser(.pdf)"
        assert repr(TextParser()) == "TextParser(.csv, .md, .txt)"
