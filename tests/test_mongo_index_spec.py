"""The two Atlas index definitions.

Every failure this file guards against is silent. Atlas will not tell you a
filter field was never declared, or that a token field was analysed into
pieces; it returns an empty result set, which reads exactly like "nothing
matched".
"""

from __future__ import annotations

import pytest

from workspace_indexer.storage.mongo_filter import exact_terms
from workspace_indexer.storage.mongo_index_spec import (
    DENSE_FIELD,
    FILTER_FIELDS,
    TEXT_FIELDS,
    text_index,
    vector_index,
)
from workspace_indexer.storage.qdrant_store import DENSE_VECTOR


def test_the_two_backends_agree_on_the_vector_field_name() -> None:
    """`Reprojector` scrolls vectors out of one store and writes them into
    another, keyed by this name. If the two drifted apart, reprojection would
    read `None` for every vector and write an empty collection -- with no
    error, because a missing key is not a failure."""
    assert DENSE_FIELD == DENSE_VECTOR


def test_the_vector_index_declares_the_dimensions_and_cosine() -> None:
    fields = vector_index(1024)["fields"]
    vector = next(f for f in fields if f["type"] == "vector")
    assert vector["path"] == DENSE_FIELD
    assert vector["numDimensions"] == 1024
    # Cosine because every measurement this project has taken is on cosine.
    # A different similarity makes the two backends' eval numbers incomparable.
    assert vector["similarity"] == "cosine"


def test_every_filterable_field_is_declared_in_the_vector_index() -> None:
    declared = {f["path"] for f in vector_index(4)["fields"] if f["type"] == "filter"}
    assert declared == set(FILTER_FIELDS)


def test_filter_fields_are_tokens_and_text_fields_are_strings() -> None:
    """A `string` mapping is analysed -- lowercased and split on punctuation --
    so `equals` against a path like `src/workspace_indexer` would match
    nothing. The search comes back empty with no error to explain it."""
    fields = text_index()["mappings"]["fields"]
    for name in FILTER_FIELDS:
        assert fields[name] == {"type": "token"}, name
    for name in TEXT_FIELDS:
        assert fields[name] == {"type": "string"}, name


def test_the_text_index_is_not_dynamic() -> None:
    """A dynamic mapping indexes every field, including the vector -- a large
    index on a shared cluster built to answer questions nobody asks."""
    assert text_index()["mappings"]["dynamic"] is False


def test_the_text_index_covers_both_halves_of_the_embedded_text() -> None:
    """Qdrant's BM25 branch encodes `embed_text`, which is the context header
    prepended to the source. We store those as two fields to avoid duplicating
    the body, so Lucene has to be pointed at both or the sparse half of hybrid
    search means something different on each backend."""
    assert "source_text" in TEXT_FIELDS
    assert "context_header" in TEXT_FIELDS


def test_every_field_the_filter_translator_uses_is_declared() -> None:
    """The translator and the index are two places one list has to be right in.

    Atlas refuses to filter on an undeclared field, so a term produced here
    that the index never declared is a search that fails or silently ignores
    the constraint.
    """
    from workspace_indexer.models import DocumentType, FileKind, SearchFilters

    every = SearchFilters(
        root_label="r",
        unit="u",
        repo_name="repo",
        language="python",
        kind=FileKind.CODE,
        path_prefix="src",
        symbol_kind="function",
        doc_type=DocumentType.NORMATIVE,
    )
    assert set(exact_terms(every)) <= set(FILTER_FIELDS)


def test_an_unknown_dtype_is_refused_rather_than_sent_to_atlas() -> None:
    with pytest.raises(ValueError, match="unsupported vector dtype"):
        vector_index(1024, dtype="int1")
