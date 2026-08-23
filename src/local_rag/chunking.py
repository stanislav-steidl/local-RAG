"""Splitting extracted documents into retrievable chunks.

Chunking is the stage that most directly determines retrieval quality, because
a chunk is simultaneously the unit that gets embedded and the unit a person
reads in an answer. Two failure modes bound the design: chunks cut mid-sentence
embed poorly and read as fragments, while chunks that are too large dilute
their own embedding until nothing matches them strongly.

The splitter is therefore *boundary-aware*. It fills up to ``chunk_size``
characters, then walks backwards to the most natural break available —
paragraph, line, sentence, clause, word — accepting a shorter chunk in exchange
for a clean edge. Only breaks in the latter part of the window are eligible: a
break close to the start would emit a chunk a fraction of the requested size
and inflate the chunk count. A hard cut therefore happens whenever that region
holds no break — because the text is an unbroken run, or because its only break
falls too early to accept.

Two invariants hold for every chunk produced here, and both are tested:

* ``end_char - start_char == len(page_content)`` — offsets always address
  exactly the stored text, so a chunk can be located in its source document.
* Chunk ends strictly increase and starts never move backwards, so no chunk
  duplicates or contains another. Consecutive chunks still overlap by roughly
  ``chunk_overlap`` characters, keeping a passage that straddles a boundary
  retrievable from at least one chunk.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import TYPE_CHECKING

from local_rag.models import Chunk, ChunkMetadata

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from local_rag.models import ExtractedDocument, PageText

__all__ = ["PAGE_SEPARATOR", "chunk_document", "iter_chunk_spans"]

#: Joins pages into a document's full text. Must match
#: :attr:`~local_rag.models.ExtractedDocument.text`, since chunk offsets are
#: expressed against that string.
PAGE_SEPARATOR = "\n\n"

#: Break candidates in descending order of preference. Earlier entries produce
#: more semantically coherent edges.
#:
#: Parsers normalise newlines to ``\n``, but this function is public and may be
#: handed raw text, so the CRLF and CR spellings of a paragraph break are listed
#: too. Without them a Windows text file would fall through to the line-break
#: search, since ``"\r\n\r\n"`` does not contain ``"\n\n"``, and a paragraph
#: boundary would silently lose to a mere line boundary.
_SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\r\n\r\n",
    "\r\r",
    "\n",
    "\r",
    ". ",
    "! ",
    "? ",
    "; ",
    ", ",
    " ",
)

#: A break is only accepted in the last portion of the window. Without a floor,
#: an early paragraph break would produce a chunk a fraction of the requested
#: size and inflate the total chunk count.
_MIN_FILL_RATIO = 0.5


def _find_break(text: str, hard_end: int, floor: int) -> int:
    """Locate the best break at or before ``hard_end``.

    Args:
        text: Text being split.
        hard_end: Exclusive offset the chunk may not extend past.
        floor: Earliest offset a break may occur at, keeping chunks from
            becoming trivially short.

    Returns:
        The offset to end the chunk at: just past the highest-priority
        separator found in ``[floor, hard_end)``, or ``hard_end`` for a hard cut
        when that range holds no separator. Note that separators before
        ``floor`` are deliberately not considered, so text can be hard cut even
        though the wider window does contain a break.
    """
    for separator in _SEPARATORS:
        found = text.rfind(separator, floor, hard_end)
        if found != -1:
            return found + len(separator)

    # The listed separators cover ASCII spacing only. Documents carry other
    # whitespace — tabs, and the non-breaking spaces Czech typography places
    # after single-letter prepositions — and a window held together by those
    # alone would otherwise be cut mid-word.
    for offset in range(hard_end - 1, floor - 1, -1):
        if text[offset].isspace():
            return offset + 1

    return hard_end


def _validate_chunk_parameters(chunk_size: int, chunk_overlap: int) -> None:
    """Check the chunking parameters, independently of any document.

    Kept separate so that validation cannot be bypassed by an early return —
    invalid settings must fail identically whether the document has text or not.

    Args:
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters shared between adjacent chunks.

    Raises:
        ValueError: If the parameters are not usable.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})"
        )


def iter_chunk_spans(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` offsets of successive chunks of ``text``.

    Leading whitespace is skipped before a window is sized, and trailing
    whitespace is trimmed from it afterwards, so the offsets address exactly the
    text a chunk stores rather than the raw window.

    Args:
        text: Text to split.
        chunk_size: Maximum characters per chunk, before whitespace trimming.
        chunk_overlap: Characters of context shared with the previous chunk.

    Yields:
        Half-open ``(start, end)`` offsets into ``text``.

    Raises:
        ValueError: If ``chunk_size`` is not positive, ``chunk_overlap`` is
            negative, or the overlap is not smaller than the chunk size. Raised
            when the function is called, not when the iterator is first
            advanced, so a caller that never consumes the result still learns
            its settings were wrong.
    """
    _validate_chunk_parameters(chunk_size, chunk_overlap)
    return _iter_spans(text, chunk_size, chunk_overlap)


def _iter_spans(text: str, chunk_size: int, chunk_overlap: int) -> Iterator[tuple[int, int]]:
    """Walk ``text`` producing chunk spans, assuming parameters are valid."""
    length = len(text)
    start = 0
    previous_end = -1

    while start < length:
        # Advance past leading whitespace *before* sizing the window. Sizing
        # from a start that points at a page separator spends part of the
        # budget on characters the chunk will not keep, pushing the remainder
        # into an extra sliver of a chunk: two 100-character pages at
        # chunk_size=100 otherwise yielded 100, 98 and 2 characters.
        while start < length and text[start].isspace():
            start += 1
        if start >= length:
            return

        hard_end = min(start + chunk_size, length)

        if hard_end == length:
            end = length
        else:
            floor = start + max(1, int(chunk_size * _MIN_FILL_RATIO))
            end = _find_break(text, hard_end, floor)

        trimmed_end = _trim_trailing(text, start, end)

        # Snapping to a boundary can make a chunk considerably shorter than
        # chunk_size. When the overlap is large relative to what a window
        # actually yields, successive windows collapse onto the same trimmed
        # span — or onto one contained in its predecessor. Requiring each span
        # to extend past the previous one keeps duplicate and redundant text
        # out of the index without ever skipping content, since starts are
        # non-decreasing.
        emitted = trimmed_end > previous_end
        if emitted:
            yield start, trimmed_end
            previous_end = trimmed_end

        if end >= length:
            return

        if emitted:
            # Guarantee forward progress. Backing off by the overlap must never
            # land at or before the current start, or the loop would not
            # terminate.
            cursor = max(end - chunk_overlap, start + 1)

            # Backing off by a fixed character count lands wherever it lands,
            # which is usually mid-word. Only chunk *endings* would then be
            # boundary-aware, and every overlapping chunk would open on a
            # fragment such as "amma delta" — embedded and cited as written.
            start = _align_to_word_start(text, cursor, max(start + 1, cursor - chunk_overlap))
        else:
            # This window added nothing beyond what is already emitted, so the
            # break search found the same separator again. Backing off by the
            # overlap would only re-derive it, shuffling forward a character at
            # a time and eventually starting a chunk mid-word. Jump clear of
            # the window instead; the content it covered is already indexed.
            start = max(end, start + 1)


def _align_to_word_start(text: str, cursor: int, lower_bound: int) -> int:
    """Move ``cursor`` back to the start of the word it landed inside.

    Moving *backwards* rather than forwards is deliberate: it widens the
    overlap slightly to take in the whole word, where moving forwards would
    discard the very context the overlap exists to provide.

    The search is bounded so that text without whitespace — an unbroken run, or
    a long identifier — does not drag the cursor far back and produce a cascade
    of near-identical chunks. When no boundary lies within reach the original
    cursor is kept: a mid-word start is unavoidable there anyway.

    Args:
        text: Text being split.
        cursor: Offset the overlap arithmetic produced.
        lower_bound: Earliest offset the search may reach.

    Returns:
        The aligned offset, never below ``lower_bound`` and never above
        ``cursor``, so forward progress and full coverage both still hold.
    """
    aligned = cursor
    while aligned > lower_bound and not text[aligned - 1].isspace():
        aligned -= 1

    starts_a_word = aligned > 0 and text[aligned - 1].isspace()
    return aligned if starts_a_word else cursor


def _trim_trailing(text: str, start: int, end: int) -> int:
    """Shrink ``end`` back past trailing whitespace.

    Only the trailing side needs handling: the caller advances past leading
    whitespace before sizing the window, so ``start`` already addresses a
    non-whitespace character.

    Trimming by adjusting the offset — rather than stripping the extracted
    string — is what keeps ``end - start == len(text[start:end])`` true for the
    stored chunk text.

    Args:
        text: Text the offsets address.
        start: Inclusive start offset, already past any whitespace.
        end: Exclusive end offset.

    Returns:
        The adjusted end offset, always greater than ``start``.
    """
    while end > start and text[end - 1].isspace():
        end -= 1
    return end


def _page_starts(pages: Sequence[PageText]) -> list[tuple[int, int]]:
    """Map each page to where its text begins in the joined document text.

    Args:
        pages: Pages in reading order.

    Returns:
        ``(start_offset, page_number)`` pairs, ordered by offset.
    """
    starts: list[tuple[int, int]] = []
    offset = 0
    for index, page in enumerate(pages):
        if index:
            offset += len(PAGE_SEPARATOR)
        starts.append((offset, page.page_number))
        offset += len(page.text)
    return starts


def _page_for_offset(page_starts: Sequence[tuple[int, int]], offset: int) -> int | None:
    """Return the page a chunk beginning at ``offset`` belongs to.

    A chunk that spans a page break is attributed to the page it *starts* on,
    which is where a reader following the citation should begin looking.

    Args:
        page_starts: Output of :func:`_page_starts`.
        offset: Offset of the chunk's first character.

    Returns:
        The page number, or ``None`` if there are no pages.
    """
    if not page_starts:
        return None

    # Binary search rather than a linear scan. Chunk count grows with page
    # count, so scanning from page 1 for every chunk would make attribution
    # quadratic in the length of the document — the cost falling hardest on the
    # long scanned PDFs this corpus is full of.
    index = bisect_right(page_starts, offset, key=lambda entry: entry[0]) - 1
    if index < 0:
        return None
    return page_starts[index][1]


def chunk_document(
    document: ExtractedDocument,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split one extracted document into retrievable chunks.

    Page numbers are attached only when the source format is genuinely
    paginated — that is, when ``page_count`` is set. For DOCX and plain text it
    stays ``None``, so citations do not claim a "page 1" that means nothing.

    Args:
        document: Document to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters of context shared between adjacent chunks.

    Returns:
        Chunks in document order. A document with no extractable text — a scan
        awaiting OCR, for instance — yields an empty list rather than an error.

    Raises:
        ValueError: If the chunking parameters are invalid. Checked before the
            document is inspected, so the outcome never depends on whether this
            particular file happened to contain text.
    """
    # Validate before the empty-document shortcut. Otherwise the same invalid
    # settings would raise for one document and pass silently for a scan
    # awaiting OCR, making the error depend on which file happened to come first.
    _validate_chunk_parameters(chunk_size, chunk_overlap)

    text = document.text
    if not text.strip():
        return []

    is_paginated = document.metadata.page_count is not None
    page_starts = _page_starts(document.pages) if is_paginated else []

    return [
        Chunk(
            page_content=text[start:end],
            metadata=ChunkMetadata(
                document=document.metadata,
                chunk_index=index,
                start_char=start,
                end_char=end,
                page_number=_page_for_offset(page_starts, start) if is_paginated else None,
            ),
        )
        for index, (start, end) in enumerate(
            iter_chunk_spans(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
    ]
