"""Tests for the shared error hierarchy.

The dual inheritance on the ingestion dependency error exists so that adding a
package-wide error type did not change what existing handlers catch. Nothing
else asserts that: the parser tests catch the ingestion-local class directly
and would keep passing if the shared base were dropped, silently breaking any
caller that spans stages.
"""

from __future__ import annotations

import pytest

from local_rag.errors import LocalRagError
from local_rag.errors import MissingDependencyError as SharedMissingDependencyError
from local_rag.ingest.base import (
    DocumentParseError,
    IngestionError,
    UnsupportedFormatError,
)
from local_rag.ingest.base import MissingDependencyError as IngestMissingDependencyError


class TestSharedHierarchy:
    def test_every_ingestion_error_is_a_package_error(self) -> None:
        """One `except LocalRagError` should cover anything this package raises."""
        assert issubclass(IngestionError, LocalRagError)

    @pytest.mark.parametrize(
        "error_type", [DocumentParseError, UnsupportedFormatError, IngestMissingDependencyError]
    )
    def test_ingestion_errors_remain_ingestion_errors(self, error_type: type[Exception]) -> None:
        assert issubclass(error_type, IngestionError)

    def test_a_missing_parser_dependency_is_catchable_as_an_ingestion_error(self) -> None:
        """Existing handlers, including load_corpus, rely on exactly this."""
        with pytest.raises(IngestionError):
            raise IngestMissingDependencyError("pdfplumber is absent")

    def test_a_missing_parser_dependency_is_catchable_as_a_shared_dependency_error(
        self,
    ) -> None:
        """A caller spanning stages should not need to know which one failed."""
        with pytest.raises(SharedMissingDependencyError):
            raise IngestMissingDependencyError("pdfplumber is absent")

    def test_the_two_dependency_errors_are_distinct_types(self) -> None:
        """The ingestion one narrows the shared one; it does not replace it."""
        assert IngestMissingDependencyError is not SharedMissingDependencyError
        assert issubclass(IngestMissingDependencyError, SharedMissingDependencyError)

    def test_the_shared_dependency_error_is_not_an_ingestion_error(self) -> None:
        """Otherwise an embedding failure would be swallowed by ingestion handlers."""
        assert not issubclass(SharedMissingDependencyError, IngestionError)
