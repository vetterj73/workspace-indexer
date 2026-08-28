"""Filter translation into Atlas's two dialects.

No server involved: these are pure functions producing query documents, and
what can be wrong about them is the shape of the document rather than anything
Mongo does with it. Wrong shapes here fail in the quietest possible way --
Atlas answers a filter it did not apply, or applies one nobody asked for, and
either way returns results that look plausible.
"""

from __future__ import annotations

from workspace_indexer.models import DocumentType, FileKind, SearchFilters
from workspace_indexer.storage.mongo_filter import match_stage, search_clauses, vector_filter


def test_no_filters_is_none_not_an_empty_document() -> None:
    """`{}` is a valid filter matching everything, so the difference is
    invisible in results and expensive in the index -- Atlas will not use the
    vector index's filter path for an empty document."""
    assert vector_filter(None) is None
    assert vector_filter(SearchFilters()) is None
    assert search_clauses(None) == ([], [])
    assert match_stage(None) == {}


def test_a_single_constraint_is_not_wrapped_in_and() -> None:
    result = vector_filter(SearchFilters(repo_name="app"))
    assert result == {"repo_name": {"$eq": "app"}}


def test_several_constraints_are_anded() -> None:
    result = vector_filter(SearchFilters(repo_name="app", language="python"))
    assert result is not None
    assert result["$and"] == [{"repo_name": {"$eq": "app"}}, {"language": {"$eq": "python"}}]


def test_a_family_of_types_is_in_not_repeated_equality() -> None:
    """`must` on two doc_types asks for a chunk that is somehow both, and
    matches nothing at all. This is the bug the Qdrant filter has a comment
    about; the same trap exists here in a different spelling."""
    result = vector_filter(SearchFilters(doc_types=[DocumentType.NORMATIVE, DocumentType.DESIGN]))
    assert result == {"doc_type": {"$in": ["normative", "design"]}}


def test_exclusions_are_nin_so_new_types_stay_included() -> None:
    """A caller saying "not tests" must not have to enumerate what it does
    want, or a type added to the taxonomy later is silently dropped."""
    result = vector_filter(SearchFilters(exclude_doc_types=[DocumentType.TEST]))
    assert result == {"doc_type": {"$nin": ["test"]}}


def test_a_directory_becomes_an_exact_match_on_ancestors() -> None:
    """`ancestors` holds every directory prefix precisely so this is an exact
    match. Neither engine can prefix-match a keyword efficiently."""
    result = vector_filter(SearchFilters(path_prefix="/src/auth/"))
    assert result == {"ancestors": {"$eq": "src/auth"}}


def test_search_constraints_are_filter_clauses_not_must() -> None:
    """A `filter` clause constrains without scoring. Under `must`, every
    surviving document would score higher for matching a constraint they all
    match, which reorders results by something the user did not ask about.
    """
    include, exclude = search_clauses(
        SearchFilters(repo_name="app", exclude_doc_types=[DocumentType.TEST])
    )
    assert include == [{"equals": {"path": "repo_name", "value": "app"}}]
    assert exclude == [{"equals": {"path": "doc_type", "value": "test"}}]


def test_match_merges_two_constraints_on_the_same_field() -> None:
    """doc_type and exclude_doc_types both target doc_type.

    Assigning each in turn would leave whichever ran last, so `find_guidance`
    narrowed to `normative` while excluding tests would silently drop one of
    the two halves of what it asked for.
    """
    match = match_stage(
        SearchFilters(doc_type=DocumentType.NORMATIVE, exclude_doc_types=[DocumentType.TEST])
    )
    assert match["doc_type"] == {"$eq": "normative", "$nin": ["test"]}


def test_match_and_vector_filter_agree_about_the_same_filters() -> None:
    """`count` uses one and `search` uses the other. If they disagreed,
    `status` would report totals no search could reproduce."""
    filters = SearchFilters(
        repo_name="app", language="python", kind=FileKind.CODE, path_prefix="src"
    )
    match = match_stage(filters)
    vector = vector_filter(filters)
    assert vector is not None
    flattened = {key: value for clause in vector["$and"] for key, value in clause.items()}
    assert flattened == match
