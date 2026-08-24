"""Filter construction.

Filters run inside the search, never after it: post-filtering a returned page
would silently shrink the result set — ask for 10 hits in one repo and get 3
because the other 7 were elsewhere.
"""

from __future__ import annotations

from qdrant_client import models

from workspace_indexer.models import FileKind, SearchFilters
from workspace_indexer.storage.qdrant_store import build_filter


def _keys(condition: models.Filter) -> set[str]:
    return {c.key for c in (condition.must or []) if isinstance(c, models.FieldCondition)}


def _value_for(condition: models.Filter, key: str) -> object:
    for c in condition.must or []:
        if isinstance(c, models.FieldCondition) and c.key == key:
            match = c.match
            return getattr(match, "value", None)
    return None


def test_no_filters_means_no_condition() -> None:
    """None, not an empty Filter: an empty must-list is a Qdrant no-op that
    still costs a filter pass."""
    assert build_filter(None) is None
    assert build_filter(SearchFilters()) is None


def test_single_field() -> None:
    condition = build_filter(SearchFilters(unit="repo_two"))
    assert condition is not None
    assert _keys(condition) == {"unit"}
    assert _value_for(condition, "unit") == "repo_two"


def test_fields_combine_as_and() -> None:
    condition = build_filter(SearchFilters(unit="repo_two", language="python"))
    assert condition is not None
    assert _keys(condition) == {"unit", "language"}


def test_enum_kind_is_sent_as_its_string_value() -> None:
    """A raw enum would serialise as its repr and match nothing."""
    condition = build_filter(SearchFilters(kind=FileKind.MARKDOWN))
    assert condition is not None
    assert _value_for(condition, "kind") == "markdown"


def test_path_prefix_matches_the_ancestors_list() -> None:
    condition = build_filter(SearchFilters(path_prefix="src/workspace_indexer"))
    assert condition is not None
    assert _keys(condition) == {"ancestors"}
    assert _value_for(condition, "ancestors") == "src/workspace_indexer"


def test_path_prefix_slashes_are_normalised() -> None:
    """`/src/pkg/` and `src/pkg` mean the same thing to a user; only one form
    is in the stored ancestors list."""
    for raw in ("/src/pkg", "src/pkg/", "/src/pkg/"):
        condition = build_filter(SearchFilters(path_prefix=raw))
        assert condition is not None
        assert _value_for(condition, "ancestors") == "src/pkg"


def test_all_filter_fields_are_honoured() -> None:
    """A field accepted by SearchFilters but dropped here would silently
    return unfiltered results."""
    filters = SearchFilters(
        root_label="src",
        unit="repo_one",
        repo_name="repo_one",
        language="python",
        kind=FileKind.CODE,
        path_prefix="src/pkg",
        symbol_kind="function",
    )
    condition = build_filter(filters)
    assert condition is not None
    assert _keys(condition) == {
        "root_label",
        "unit",
        "repo_name",
        "language",
        "kind",
        "ancestors",
        "symbol_kind",
    }


def test_is_empty_reflects_only_set_fields() -> None:
    assert SearchFilters().is_empty()
    assert not SearchFilters(language="python").is_empty()
