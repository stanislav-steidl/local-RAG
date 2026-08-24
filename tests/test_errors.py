"""Tests for the shared error hierarchy.

The dual inheritance on the ingestion dependency error exists so that adding a
package-wide error type did not change what existing handlers catch. Nothing
else asserts that: the parser tests catch the ingestion-local class directly
and would keep passing if the shared base were dropped, silently breaking any
caller that spans stages.
"""

from __future__ import annotations

import pytest

from local_rag.errors import LocalRagError, optional_dependency
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


class TestOptionalDependency:
    """The missing-versus-broken distinction, in the one place that now makes it."""

    def test_a_successful_import_passes_through(self) -> None:
        with optional_dependency("json", "install json somehow"):
            pass

    def test_the_named_module_being_absent_reports_the_extra(self) -> None:
        with (
            pytest.raises(SharedMissingDependencyError, match="install the thing"),
            optional_dependency("thing", "install the thing"),
        ):
            raise ModuleNotFoundError("No module named 'thing'", name="thing")

    def test_a_submodule_counts_as_the_module(self) -> None:
        """The extra is unusable whether the package or a piece of it is missing."""
        with (
            pytest.raises(SharedMissingDependencyError, match="install the thing"),
            optional_dependency("thing", "install the thing"),
        ):
            raise ModuleNotFoundError("gone", name="thing.inner")

    def test_a_different_missing_module_propagates_untouched(self) -> None:
        """An installed package failing its own imports is not an absent extra.

        Translating this would send someone to reinstall something already
        present, and hide the transitive module that actually went missing.
        """
        with (
            pytest.raises(ModuleNotFoundError, match="peft") as excinfo,
            optional_dependency("thing", "install the thing"),
        ):
            raise ModuleNotFoundError("No module named 'peft'", name="peft")

        assert not isinstance(excinfo.value, SharedMissingDependencyError)

    def test_an_unnamed_module_error_propagates(self) -> None:
        """Without a name there is nothing to attribute the failure to."""
        with (
            pytest.raises(ModuleNotFoundError),
            optional_dependency("thing", "install the thing"),
        ):
            raise ModuleNotFoundError("something went wrong")

    def test_other_exceptions_are_untouched(self) -> None:
        with (
            pytest.raises(RuntimeError, match="unrelated"),
            optional_dependency("thing", "install the thing"),
        ):
            raise RuntimeError("unrelated")

    def test_a_stage_specific_error_type_is_honoured(self) -> None:
        """Ingestion passes its own subclass so IngestionError handlers still fire."""
        with (
            pytest.raises(IngestionError),
            optional_dependency(
                "pdfplumber", "install parsing", error_type=IngestMissingDependencyError
            ),
        ):
            raise ModuleNotFoundError("nope", name="pdfplumber")
