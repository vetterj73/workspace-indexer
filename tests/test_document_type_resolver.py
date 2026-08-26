"""Turning a caller's word into a document type.

The tests that matter here are all about the *failure* path. A resolver that
returns None or an empty filter on a bad input is how an agent concludes a
workspace has no specifications when it merely said "spec".
"""

from __future__ import annotations

import pytest

from workspace_indexer.mcp import ALIASES, DocumentTypeResolver, UnknownDocumentTypeError
from workspace_indexer.models import DocumentType


@pytest.fixture
def resolver() -> DocumentTypeResolver:
    return DocumentTypeResolver()


def test_exact_names_resolve(resolver: DocumentTypeResolver) -> None:
    for doc_type in DocumentType:
        assert resolver.resolve(doc_type.value) is doc_type


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("spec", DocumentType.NORMATIVE),
        ("ADR", DocumentType.NORMATIVE),
        ("  Specification  ", DocumentType.NORMATIVE),
        ("architecture", DocumentType.DESIGN),
        ("readme", DocumentType.GUIDE),
        ("changelog", DocumentType.RECORD),
        ("code", DocumentType.IMPLEMENTATION),
    ],
)
def test_near_misses_just_work(
    resolver: DocumentTypeResolver, given: str, expected: DocumentType
) -> None:
    """The agent is guessing our vocabulary from a one-line description.
    Being right about the intent and wrong about the word is the common case."""
    assert resolver.resolve(given) is expected


def test_case_and_hyphens_are_normalised(resolver: DocumentTypeResolver) -> None:
    assert resolver.resolve("NORMATIVE") is DocumentType.NORMATIVE
    assert resolver.resolve("Design") is DocumentType.DESIGN


def test_unknown_type_raises_rather_than_matching_nothing(
    resolver: DocumentTypeResolver,
) -> None:
    """The single worst failure this server can have: a silent empty result
    that reads as "this workspace has no guidance"."""
    with pytest.raises(UnknownDocumentTypeError) as caught:
        resolver.resolve("blueprint")

    message = str(caught.value)
    assert "blueprint" in message
    # The error has to be actionable on its own -- the agent should not need a
    # second round trip to discover the vocabulary.
    assert "normative" in message
    assert "spec" in message


def test_error_lists_every_valid_type(resolver: DocumentTypeResolver) -> None:
    with pytest.raises(UnknownDocumentTypeError) as caught:
        resolver.resolve("nonsense")
    for doc_type in DocumentType:
        assert doc_type.value in str(caught.value)


def test_resolve_all_reports_the_first_bad_entry(resolver: DocumentTypeResolver) -> None:
    with pytest.raises(UnknownDocumentTypeError, match="bogus"):
        resolver.resolve_all(["normative", "bogus", "design"])


def test_every_alias_points_at_a_real_type() -> None:
    """An alias for a renamed type would resolve to nothing and be worse than
    no alias at all."""
    for target in ALIASES.values():
        assert target in set(DocumentType)


def test_aliases_do_not_shadow_real_names() -> None:
    """A real type name must never be reachable only through the alias map,
    or renaming the alias would silently change what the name means."""
    assert not {t.value for t in DocumentType} & set(ALIASES)
