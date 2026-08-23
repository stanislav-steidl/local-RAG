"""Tests for boundary-aware chunking."""

from __future__ import annotations

from itertools import pairwise

import pytest

from local_rag.chunking import PAGE_SEPARATOR, chunk_document, iter_chunk_spans
from local_rag.models import ExtractedDocument, PageText

from .conftest import make_document_metadata


def spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Chunk spans as a list, for convenient assertions."""
    return list(iter_chunk_spans(text, chunk_size=size, chunk_overlap=overlap))


def texts(text: str, size: int, overlap: int) -> list[str]:
    """The chunk texts a split produces."""
    return [text[start:end] for start, end in spans(text, size, overlap)]


def make_document(
    pages: list[str],
    *,
    paginated: bool = True,
) -> ExtractedDocument:
    """Build an ExtractedDocument from page texts."""
    return ExtractedDocument(
        metadata=make_document_metadata(page_count=len(pages) if paginated else None),
        pages=tuple(
            PageText(page_number=number, text=text) for number, text in enumerate(pages, start=1)
        ),
    )


class TestParameterValidation:
    def test_non_positive_chunk_size_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            spans("text", 0, 0)

    def test_negative_overlap_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            spans("text", 10, -1)

    def test_overlap_equal_to_size_is_rejected(self) -> None:
        """Equal values would leave the cursor unable to advance."""
        with pytest.raises(ValueError, match="must be smaller than"):
            spans("text", 10, 10)

    def test_overlap_larger_than_size_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be smaller than"):
            spans("text", 10, 20)

    def test_validation_happens_on_call_not_on_first_iteration(self) -> None:
        """A generator would defer validation until consumed, hiding bad settings."""
        with pytest.raises(ValueError, match="must be smaller than"):
            iter_chunk_spans("text", chunk_size=10, chunk_overlap=10)


class TestInvariants:
    @pytest.mark.parametrize("size", [16, 40, 128])
    @pytest.mark.parametrize("overlap", [0, 5, 15])
    def test_offsets_address_exactly_the_stored_text(self, size: int, overlap: int) -> None:
        """Offsets must equal the chunk length, or citations cannot be located."""
        text = "Some sentence here. " * 40
        for start, end in spans(text, size, overlap):
            assert end - start == len(text[start:end])

    @pytest.mark.parametrize("size", [16, 40, 128])
    @pytest.mark.parametrize("overlap", [0, 5, 15])
    def test_spans_advance_and_stay_in_bounds(self, size: int, overlap: int) -> None:
        text = "Some sentence here. " * 40
        previous_start, previous_end = -1, -1
        for start, end in spans(text, size, overlap):
            assert 0 <= start < end <= len(text)
            assert start >= previous_start
            assert end > previous_end
            previous_start, previous_end = start, end

    @pytest.mark.parametrize("overlap", [0, 5, 15])
    def test_no_chunk_duplicates_or_contains_another(self, overlap: int) -> None:
        """A large overlap once collapsed successive windows onto the same span.

        Emitting those would put byte-identical text in the index twice, so
        each span must extend past its predecessor.
        """
        text = "Some sentence here. " * 40
        produced = spans(text, 16, overlap)

        assert len(produced) == len(set(produced))
        for (first_start, first_end), (second_start, second_end) in pairwise(produced):
            contained = second_start >= first_start and second_end <= first_end
            assert not contained

    @pytest.mark.parametrize("size", [16, 40, 128])
    def test_every_character_of_content_is_covered(self, size: int) -> None:
        """No non-whitespace character may be lost between chunks."""
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
        covered: set[int] = set()
        for start, end in spans(text, size, 0):
            covered.update(range(start, end))

        missing = {
            index for index, char in enumerate(text) if not char.isspace() and index not in covered
        }
        assert missing == set()

    def test_chunks_never_exceed_the_requested_size(self) -> None:
        text = "word " * 200
        assert all(end - start <= 40 for start, end in spans(text, 40, 10))

    def test_no_chunk_has_surrounding_whitespace(self) -> None:
        text = "First paragraph.\n\n\n   Second paragraph.\n\n\nThird paragraph."
        assert all(chunk == chunk.strip() for chunk in texts(text, 20, 5))


class TestBoundaryAwareness:
    def test_prefers_a_paragraph_break(self) -> None:
        text = "First paragraph here.\n\nSecond paragraph follows on after."
        assert texts(text, 40, 0)[0] == "First paragraph here."

    def test_falls_back_to_a_line_break(self) -> None:
        text = "First line of text here\nSecond line of text follows here"
        assert texts(text, 40, 0)[0] == "First line of text here"

    def test_falls_back_to_a_sentence_boundary(self) -> None:
        text = "The first sentence ends here. The second sentence continues onwards."
        assert texts(text, 40, 0)[0] == "The first sentence ends here."

    def test_falls_back_to_a_word_boundary(self) -> None:
        """Splitting mid-word would embed a fragment that matches nothing."""
        text = "alpha beta gamma delta epsilon zeta eta theta"
        assert not texts(text, 20, 0)[0].endswith(("alph", "bet", "gamm"))
        assert texts(text, 20, 0)[0] == "alpha beta gamma"

    def test_hard_cuts_an_unbroken_run(self) -> None:
        """Text with no separator anywhere has no natural edge to snap to."""
        text = "x" * 50
        assert [end - start for start, end in spans(text, 20, 0)] == [20, 20, 10]

    def test_a_break_too_early_in_the_window_is_not_taken(self) -> None:
        """Honouring it would emit a tiny chunk and inflate the chunk count."""
        text = "Hi.\n\n" + "continuous text without any breaks at all here"
        assert len(texts(text, 40, 0)[0]) > 20

    def test_hard_cuts_when_the_only_break_falls_before_the_accepted_region(self) -> None:
        """A break exists, yet the chunk is still cut hard — by design.

        Only the latter part of the window is eligible, so a boundary near the
        start is passed over. Worth pinning down because it contradicts the
        intuitive reading of "hard cuts happen only without a boundary".
        """
        text = "Hi.\n\n" + "x" * 60

        first = texts(text, 20, 0)[0]

        assert first == "Hi.\n\n" + "x" * 15
        assert first.endswith("x")


class TestOverlap:
    def test_consecutive_chunks_share_context(self) -> None:
        """A passage straddling a boundary must survive in at least one chunk."""
        text = "word " * 100
        produced = spans(text, 50, 20)

        assert len(produced) > 1
        for (_, first_end), (second_start, _) in pairwise(produced):
            assert second_start < first_end

    def test_zero_overlap_produces_disjoint_chunks(self) -> None:
        text = "word " * 100
        produced = spans(text, 50, 0)

        for (_, first_end), (second_start, _) in pairwise(produced):
            assert second_start >= first_end

    def test_terminates_when_overlap_would_prevent_progress(self) -> None:
        """A pathological break near the window start must not loop forever."""
        text = ("a" * 3 + " ") * 60
        assert len(spans(text, 8, 7)) < 1000


class TestDegenerateInput:
    def test_empty_text_yields_nothing(self) -> None:
        assert spans("", 100, 10) == []

    def test_whitespace_only_text_yields_nothing(self) -> None:
        assert spans("   \n\n\t  ", 100, 10) == []

    def test_text_shorter_than_a_chunk_is_a_single_chunk(self) -> None:
        assert texts("short text", 100, 10) == ["short text"]

    def test_text_exactly_one_chunk_long(self) -> None:
        text = "x" * 20
        assert texts(text, 20, 5) == [text]


class TestChunkDocument:
    def test_produces_chunks_in_order_with_sequential_indices(self) -> None:
        document = make_document(["Some sentence here. " * 20])

        chunks = chunk_document(document, chunk_size=60, chunk_overlap=10)

        assert len(chunks) > 1
        assert [chunk.metadata.chunk_index for chunk in chunks] == list(range(len(chunks)))

    def test_chunks_carry_the_source_document_metadata(self) -> None:
        document = make_document(["Some text to split into pieces."])

        chunk = chunk_document(document, chunk_size=20, chunk_overlap=5)[0]

        assert chunk.metadata.document is document.metadata

    def test_offsets_locate_the_chunk_in_the_document_text(self) -> None:
        document = make_document(["Some sentence here. " * 20])

        for chunk in chunk_document(document, chunk_size=60, chunk_overlap=10):
            start, end = chunk.metadata.start_char, chunk.metadata.end_char
            assert document.text[start:end] == chunk.page_content

    def test_empty_document_yields_no_chunks(self) -> None:
        assert chunk_document(make_document([]), chunk_size=100, chunk_overlap=10) == []

    def test_scanned_document_yields_no_chunks(self) -> None:
        """A page with no text layer is awaiting OCR, not an error."""
        document = make_document(["", "   ", "\n"])

        assert chunk_document(document, chunk_size=100, chunk_overlap=10) == []

    def test_invalid_parameters_propagate(self) -> None:
        document = make_document(["some text"])

        with pytest.raises(ValueError, match="must be smaller than"):
            chunk_document(document, chunk_size=10, chunk_overlap=10)

    @pytest.mark.parametrize(
        ("pages", "description"),
        [([], "no pages"), (["", "   "], "scanned pages")],
    )
    def test_invalid_parameters_are_rejected_even_without_text(
        self, pages: list[str], description: str
    ) -> None:
        """Validation must not depend on the document happening to have content.

        The empty-document shortcut once returned before any check ran, so the
        same misconfiguration raised for a text document but passed silently
        for a scan — making the failure depend on corpus ordering.
        """
        document = make_document(pages)

        with pytest.raises(ValueError, match="must be smaller than"):
            chunk_document(document, chunk_size=10, chunk_overlap=10)


class TestPageAttribution:
    def test_chunks_are_attributed_to_the_page_they_start_on(self) -> None:
        document = make_document(["A" * 100, "B" * 100, "C" * 100])

        chunks = chunk_document(document, chunk_size=100, chunk_overlap=0)

        for chunk in chunks:
            first_letter = chunk.page_content[0]
            expected = {"A": 1, "B": 2, "C": 3}[first_letter]
            assert chunk.metadata.page_number == expected

    def test_a_chunk_spanning_a_break_reports_its_starting_page(self) -> None:
        """That is where a reader following the citation should begin looking."""
        document = make_document(["A" * 30, "B" * 30])

        chunk = chunk_document(document, chunk_size=100, chunk_overlap=0)[0]

        assert PAGE_SEPARATOR in chunk.page_content
        assert chunk.metadata.page_number == 1

    def test_unpaginated_documents_get_no_page_number(self) -> None:
        """Otherwise every DOCX citation would claim a meaningless 'page 1'."""
        document = make_document(["Some text here.", "More text here."], paginated=False)

        chunks = chunk_document(document, chunk_size=20, chunk_overlap=0)

        assert chunks
        assert all(chunk.metadata.page_number is None for chunk in chunks)

    def test_citation_renders_the_page_for_paginated_sources(self) -> None:
        document = make_document(["A" * 100, "B" * 100])

        chunks = chunk_document(document, chunk_size=100, chunk_overlap=0)

        assert chunks[-1].citation == "contracts/lease.pdf (p. 2)"

    def test_citation_omits_the_page_for_unpaginated_sources(self) -> None:
        document = make_document(["Some text here."], paginated=False)

        chunk = chunk_document(document, chunk_size=100, chunk_overlap=0)[0]

        assert chunk.citation == "contracts/lease.pdf"

    def test_pages_with_no_text_do_not_misattribute_later_chunks(self) -> None:
        """An empty page still occupies an offset range in the joined text."""
        document = make_document(["A" * 50, "", "C" * 50])

        chunks = chunk_document(document, chunk_size=50, chunk_overlap=0)

        assert chunks[-1].page_content.startswith("C")
        assert chunks[-1].metadata.page_number == 3
