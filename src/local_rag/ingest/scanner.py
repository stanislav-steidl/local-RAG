"""Discovery of source files in the corpus, and the metadata derivable from disk.

The scanner answers "what is in the corpus, and has it changed?" without
opening a single document. It is deliberately format-agnostic: deciding what a
file *contains* belongs to a parser.

A corpus is a real user directory, so the scanner is written to tolerate what
such directories actually contain — Office lock files, OS metadata droppings,
permission-denied entries and files that vanish mid-scan. Anything it cannot
read is skipped with a warning rather than aborting a run that may be part-way
through several gigabytes.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from local_rag.models import DocumentMetadata, SourceType

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator

__all__ = [
    "IGNORED_FILE_NAMES",
    "IGNORED_NAME_PREFIXES",
    "IMAGE_EXTENSIONS",
    "build_document_metadata",
    "compute_content_hash",
    "iter_source_files",
    "scan_corpus",
]

logger = logging.getLogger(__name__)

#: Read size for hashing. Large enough to keep syscall overhead negligible,
#: small enough that a multi-gigabyte file never lands in memory.
_HASH_BLOCK_SIZE = 1024 * 1024

#: Extensions treated as photographs rather than documents.
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".heic"})

#: Operating-system and cloud-sync bookkeeping files, never user content.
IGNORED_FILE_NAMES = frozenset({"thumbs.db", "desktop.ini", ".ds_store", "icon\r"})

#: ``~$`` prefixes Microsoft Office lock files, which are present whenever a
#: document is open and contain no readable content.
IGNORED_NAME_PREFIXES = ("~$", ".~lock.")


def compute_content_hash(path: Path) -> str:
    """Return the SHA-256 digest of a file's contents, read incrementally.

    The digest is what lets re-indexing skip files that have not actually
    changed. It is read in blocks so that file size does not bound memory.

    Note:
        Hashing every file costs a full read of the corpus. Incremental
        re-indexing will gate this on cheaper ``(size, mtime)`` comparisons and
        hash only the candidates that appear to have changed.

    Args:
        path: File to digest.

    Returns:
        The digest as a lowercase hexadecimal string.

    Raises:
        OSError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_HASH_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def classify_source_type(extension: str) -> SourceType:
    """Classify a file as a photo or a document from its extension.

    Args:
        extension: File extension including the leading dot, any case.

    Returns:
        :attr:`SourceType.PHOTO` for known image extensions, otherwise
        :attr:`SourceType.DOCUMENT`.
    """
    return SourceType.PHOTO if extension.lower() in IMAGE_EXTENSIONS else SourceType.DOCUMENT


def _is_ignored(name: str) -> bool:
    """Whether a file or directory name is bookkeeping rather than content."""
    lowered = name.lower()
    if lowered in IGNORED_FILE_NAMES:
        return True
    if name.startswith("."):
        # Hidden entries: dotfiles, and tool directories such as .git or .lancedb.
        return True
    return any(lowered.startswith(prefix) for prefix in IGNORED_NAME_PREFIXES)


def iter_source_files(
    root: Path,
    *,
    extensions: Collection[str] | None = None,
) -> Iterator[Path]:
    """Yield candidate source files beneath ``root``, in a deterministic order.

    Hidden entries are pruned at the directory level, so a nested ``.git`` or
    index directory costs nothing to skip. Symlinks are not followed and
    symlinked files are skipped, which keeps a cycle from turning a scan into
    an infinite walk and stops the same document being indexed twice.

    Args:
        root: Corpus root to walk.
        extensions: If given, only files whose extension appears here are
            yielded. Matching is case-insensitive and each entry must include
            the leading dot.

    Yields:
        Paths to candidate files. The walk is top-down, so a directory's own
        files precede those of its subdirectories, and entries are sorted
        within each directory. The resulting order is therefore not globally
        lexicographic, but it is stable: repeated scans of an unchanged corpus
        yield exactly the same sequence.
    """
    wanted = {ext.lower() for ext in extensions} if extensions is not None else None

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Prune in place so os.walk does not descend into ignored directories.
        dirnames[:] = sorted(name for name in dirnames if not _is_ignored(name))

        for filename in sorted(filenames):
            if _is_ignored(filename):
                continue

            path = Path(dirpath) / filename
            if wanted is not None and path.suffix.lower() not in wanted:
                continue
            if path.is_symlink():
                logger.debug("Skipping symlink: %s", path)
                continue

            yield path


def build_document_metadata(path: Path, root: Path) -> DocumentMetadata:
    """Describe a single file using only what the filesystem reports.

    ``page_count`` is left unset: pagination is a property of the file's
    contents, which only a parser can determine.

    Args:
        path: File to describe.
        root: Corpus root that ``path`` lies beneath, used to derive the
            portable relative path stored in the index.

    Returns:
        Metadata describing the file.

    Raises:
        OSError: If the file cannot be read or inspected.
        ValueError: If ``path`` is not located beneath ``root``.
    """
    stat_result = path.stat()
    return DocumentMetadata(
        relative_path=path.relative_to(root).as_posix(),
        source_type=classify_source_type(path.suffix),
        file_extension=path.suffix.lower(),
        size_bytes=stat_result.st_size,
        modified_at=datetime.fromtimestamp(stat_result.st_mtime, tz=timezone.utc),
        content_hash=compute_content_hash(path),
    )


def scan_corpus(
    root: Path,
    *,
    extensions: Collection[str] | None = None,
) -> Iterator[DocumentMetadata]:
    """Walk the corpus and describe every file that can be read.

    Files that cannot be inspected — removed mid-scan, permission denied, or an
    unreadable cloud placeholder — are logged and skipped. A single unreadable
    file must not abort an indexing run over a large corpus.

    Args:
        root: Corpus root to scan.
        extensions: Optional case-insensitive extension allow-list, each entry
            including the leading dot.

    Yields:
        Metadata for each readable file, in deterministic order.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
        NotADirectoryError: If ``root`` is not a directory.
    """
    if not root.exists():
        raise FileNotFoundError(f"corpus root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"corpus root is not a directory: {root}")

    for path in iter_source_files(root, extensions=extensions):
        try:
            yield build_document_metadata(path, root)
        except OSError as error:
            logger.warning("Skipping unreadable file %s: %s", path, error)
