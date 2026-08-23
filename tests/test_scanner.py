"""Tests for corpus discovery and filesystem-derived metadata."""

from __future__ import annotations

import hashlib
import logging
from datetime import timezone
from pathlib import Path

import pytest

from local_rag.ingest import scanner
from local_rag.ingest.scanner import (
    build_document_metadata,
    classify_source_type,
    compute_content_hash,
    iter_source_files,
    scan_corpus,
)
from local_rag.models import SourceType


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """An empty corpus root."""
    root = tmp_path / "corpus"
    root.mkdir()
    return root


def write(root: Path, relative: str, content: str = "content") -> Path:
    """Create a file (and any parent directories) under ``root``."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def relative_names(root: Path, **kwargs: object) -> list[str]:
    """Discovered files as corpus-relative POSIX paths."""
    return [
        path.relative_to(root).as_posix()
        for path in iter_source_files(root, **kwargs)  # type: ignore[arg-type]
    ]


class TestComputeContentHash:
    def test_matches_a_known_digest(self, corpus: Path) -> None:
        path = write(corpus, "a.txt", "hello")
        assert compute_content_hash(path) == hashlib.sha256(b"hello").hexdigest()

    def test_identical_content_hashes_identically(self, corpus: Path) -> None:
        """Two copies of a document must not both be indexed."""
        first = write(corpus, "a.txt", "same")
        second = write(corpus, "nested/b.txt", "same")
        assert compute_content_hash(first) == compute_content_hash(second)

    def test_differing_content_hashes_differently(self, corpus: Path) -> None:
        assert compute_content_hash(write(corpus, "a.txt", "one")) != compute_content_hash(
            write(corpus, "b.txt", "two")
        )

    def test_empty_file_is_hashable(self, corpus: Path) -> None:
        path = write(corpus, "empty.txt", "")
        assert compute_content_hash(path) == hashlib.sha256(b"").hexdigest()

    def test_content_spanning_multiple_blocks(
        self, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exercise the read loop rather than trusting a single-block read."""
        monkeypatch.setattr(scanner, "_HASH_BLOCK_SIZE", 4)
        payload = "abcdefghijklmnopqrstuvwxyz"
        path = write(corpus, "long.txt", payload)
        assert compute_content_hash(path) == hashlib.sha256(payload.encode()).hexdigest()

    def test_missing_file_raises(self, corpus: Path) -> None:
        with pytest.raises(FileNotFoundError):
            compute_content_hash(corpus / "absent.txt")


class TestClassifySourceType:
    @pytest.mark.parametrize("extension", [".jpg", ".JPEG", ".png", ".HEIC", ".tiff"])
    def test_image_extensions_are_photos(self, extension: str) -> None:
        assert classify_source_type(extension) is SourceType.PHOTO

    @pytest.mark.parametrize("extension", [".pdf", ".DOCX", ".txt", ".xml", ""])
    def test_everything_else_is_a_document(self, extension: str) -> None:
        assert classify_source_type(extension) is SourceType.DOCUMENT


class TestIterSourceFiles:
    def test_finds_files_at_every_depth(self, corpus: Path) -> None:
        write(corpus, "top.pdf")
        write(corpus, "contracts/lease.pdf")
        write(corpus, "contracts/2022/addendum.pdf")

        # Top-down walk: a directory's own files precede its subdirectories.
        assert relative_names(corpus) == [
            "top.pdf",
            "contracts/lease.pdf",
            "contracts/2022/addendum.pdf",
        ]

    def test_ordering_is_deterministic(self, corpus: Path) -> None:
        """Repeated scans must agree, or incremental indexing cannot be reasoned about."""
        for name in ("c.pdf", "a.pdf", "b.pdf", "sub/z.pdf", "sub/y.pdf"):
            write(corpus, name)

        assert relative_names(corpus) == relative_names(corpus)
        assert relative_names(corpus) == ["a.pdf", "b.pdf", "c.pdf", "sub/y.pdf", "sub/z.pdf"]

    def test_empty_corpus_yields_nothing(self, corpus: Path) -> None:
        assert relative_names(corpus) == []

    def test_hidden_directories_are_pruned(self, corpus: Path) -> None:
        """A nested .git or index directory must cost nothing to skip."""
        write(corpus, "keep.pdf")
        write(corpus, ".git/objects/blob")
        write(corpus, ".lancedb/data.lance")

        assert relative_names(corpus) == ["keep.pdf"]

    def test_hidden_files_are_skipped(self, corpus: Path) -> None:
        write(corpus, "keep.pdf")
        write(corpus, ".env")

        assert relative_names(corpus) == ["keep.pdf"]

    def test_office_lock_files_are_skipped(self, corpus: Path) -> None:
        """`~$` files appear whenever a document is open and hold no content."""
        write(corpus, "report.docx")
        write(corpus, "~$report.docx")

        assert relative_names(corpus) == ["report.docx"]

    @pytest.mark.parametrize("name", ["Thumbs.db", "thumbs.db", "desktop.ini", ".DS_Store"])
    def test_operating_system_metadata_is_skipped(self, corpus: Path, name: str) -> None:
        write(corpus, "keep.pdf")
        write(corpus, name)

        assert relative_names(corpus) == ["keep.pdf"]

    def test_extension_filter_is_applied(self, corpus: Path) -> None:
        write(corpus, "a.pdf")
        write(corpus, "b.docx")
        write(corpus, "c.zip")

        assert relative_names(corpus, extensions={".pdf", ".docx"}) == ["a.pdf", "b.docx"]

    def test_extension_filter_ignores_case(self, corpus: Path) -> None:
        """Windows corpora routinely mix .PDF and .pdf."""
        write(corpus, "shouty.PDF")

        assert relative_names(corpus, extensions={".pdf"}) == ["shouty.PDF"]
        assert relative_names(corpus, extensions={".PDF"}) == ["shouty.PDF"]

    def test_no_filter_yields_every_extension(self, corpus: Path) -> None:
        write(corpus, "a.pdf")
        write(corpus, "b.unknown")

        assert relative_names(corpus) == ["a.pdf", "b.unknown"]

    def test_symlinked_files_are_skipped(self, corpus: Path) -> None:
        """Following links would index the same document twice."""
        target = write(corpus, "real.pdf")
        link = corpus / "alias.pdf"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("creating symlinks requires privileges unavailable here")

        assert relative_names(corpus) == ["real.pdf"]


class TestBuildDocumentMetadata:
    def test_records_filesystem_facts(self, corpus: Path) -> None:
        path = write(corpus, "contracts/lease.pdf", "body")

        meta = build_document_metadata(path, corpus)

        assert meta.relative_path == "contracts/lease.pdf"
        assert meta.file_name == "lease.pdf"
        assert meta.file_extension == ".pdf"
        assert meta.size_bytes == len(b"body")
        assert meta.content_hash == hashlib.sha256(b"body").hexdigest()
        assert meta.source_type is SourceType.DOCUMENT

    def test_relative_path_uses_forward_slashes(self, corpus: Path) -> None:
        """The index must read identically on the platform it was not built on."""
        path = write(corpus, "a/b/c.pdf")

        assert "\\" not in build_document_metadata(path, corpus).relative_path

    def test_modified_at_is_timezone_aware(self, corpus: Path) -> None:
        path = write(corpus, "a.pdf")

        assert build_document_metadata(path, corpus).modified_at.tzinfo is timezone.utc

    def test_extension_is_normalised_to_lowercase(self, corpus: Path) -> None:
        path = write(corpus, "shouty.PDF")

        assert build_document_metadata(path, corpus).file_extension == ".pdf"

    def test_page_count_is_left_for_a_parser_to_determine(self, corpus: Path) -> None:
        """Pagination is a property of contents, which the scanner never reads."""
        path = write(corpus, "a.pdf")

        assert build_document_metadata(path, corpus).page_count is None

    def test_images_are_classified_as_photos(self, corpus: Path) -> None:
        path = write(corpus, "album/holiday.jpg")

        assert build_document_metadata(path, corpus).source_type is SourceType.PHOTO

    def test_path_outside_the_root_is_rejected(self, corpus: Path, tmp_path: Path) -> None:
        outside = tmp_path / "elsewhere.pdf"
        outside.write_text("x", encoding="utf-8")

        with pytest.raises(ValueError, match=r"elsewhere\.pdf"):
            build_document_metadata(outside, corpus)

    def test_outside_path_is_rejected_before_the_filesystem_is_touched(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        """Containment is checked first, so the error never depends on the file existing.

        Statting first would report OSError for a non-existent outside path
        instead of the documented ValueError, and would disclose whether a path
        beyond the corpus exists.
        """
        with pytest.raises(ValueError, match=r"absent\.pdf"):
            build_document_metadata(tmp_path / "absent.pdf", corpus)


class TestScanCorpus:
    def test_describes_every_file(self, corpus: Path) -> None:
        write(corpus, "a.pdf", "one")
        write(corpus, "nested/b.docx", "two")

        results = list(scan_corpus(corpus))

        assert [meta.relative_path for meta in results] == ["a.pdf", "nested/b.docx"]

    def test_applies_the_extension_filter(self, corpus: Path) -> None:
        write(corpus, "a.pdf")
        write(corpus, "b.zip")

        results = list(scan_corpus(corpus, extensions={".pdf"}))

        assert [meta.relative_path for meta in results] == ["a.pdf"]

    def test_missing_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            list(scan_corpus(tmp_path / "absent"))

    def test_root_that_is_a_file_raises(self, corpus: Path) -> None:
        path = write(corpus, "a.pdf")

        with pytest.raises(NotADirectoryError, match="not a directory"):
            list(scan_corpus(path))

    def test_unreadable_files_are_skipped_not_fatal(
        self, corpus: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One permission-denied file must not abort a multi-gigabyte scan."""
        write(corpus, "readable.pdf")
        write(corpus, "locked.pdf")
        original = scanner.build_document_metadata

        def fail_on_locked(path: Path, root: Path) -> object:
            if path.name == "locked.pdf":
                raise PermissionError(13, "Permission denied")
            return original(path, root)

        monkeypatch.setattr(scanner, "build_document_metadata", fail_on_locked)

        with caplog.at_level(logging.WARNING):
            results = list(scan_corpus(corpus))

        assert [meta.relative_path for meta in results] == ["readable.pdf"]
        assert "locked.pdf" in caplog.text

    def test_is_lazy(self, corpus: Path) -> None:
        """Scanning returns a generator so a huge corpus is never materialised."""
        write(corpus, "a.pdf")

        assert not isinstance(scan_corpus(corpus), list)
        assert next(iter(scan_corpus(corpus))).relative_path == "a.pdf"

    def test_walks_a_realistic_messy_corpus(self, corpus: Path) -> None:
        """The shape an actual synced documents folder takes."""
        write(corpus, "Smlouva 2022.pdf")
        write(corpus, "faktury/faktura_01.pdf")
        write(corpus, "faktury/~$faktura_01.docx")
        write(corpus, "album/dovolená.jpg")
        write(corpus, "Thumbs.db")
        write(corpus, ".git/config")

        results = list(scan_corpus(corpus))

        assert [meta.relative_path for meta in results] == [
            "Smlouva 2022.pdf",
            "album/dovolená.jpg",
            "faktury/faktura_01.pdf",
        ]
        assert results[1].source_type is SourceType.PHOTO


def test_scanner_does_not_follow_directory_symlinks(corpus: Path) -> None:
    """A symlink loop must not turn a scan into an infinite walk."""
    write(corpus, "real/a.pdf")
    link = corpus / "loop"
    try:
        link.symlink_to(corpus, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("creating symlinks requires privileges unavailable here")

    assert [path.relative_to(corpus).as_posix() for path in iter_source_files(corpus)] == [
        "real/a.pdf"
    ]
