"""Translating SearchFilters into the two filter dialects Atlas uses.

Two, not one, and that is Atlas's doing rather than ours. `$vectorSearch` takes
a subset of MQL in its `filter` field; `$search` takes Lucene clauses inside a
`compound` operator. The same restriction applies to both: a field can only be
filtered on if the corresponding index declared it, which is why
`mongo_index_spec.FILTER_FIELDS` is the single list both indexes are built from
and both functions here draw on.

Pre-filters in both cases, never a `$match` after the search. Post-filtering a
returned page silently shrinks the result set -- ask for 10 hits in one repo
and get 3 because the other 7 were elsewhere -- and Atlas makes that mistake
especially easy to write, because `$match` after `$search` is valid and looks
fine until you count the results.
"""

from __future__ import annotations

from typing import Any

from workspace_indexer.models import SearchFilters


def exact_terms(filters: SearchFilters) -> dict[str, str]:
    """The one-value-per-field constraints, in payload terms.

    Public because it is half of an invariant: every key it can produce must
    be declared in `mongo_index_spec.FILTER_FIELDS`, and a test asserts that.
    Atlas refuses to filter on an undeclared field, so a key that drifts out of
    one list is a search that fails or quietly ignores the constraint.
    """
    candidates = {
        "root_label": filters.root_label,
        "unit": filters.unit,
        "repo_name": filters.repo_name,
        "language": filters.language,
        "symbol_kind": filters.symbol_kind,
        "kind": filters.kind.value if filters.kind else None,
        "doc_type": filters.doc_type.value if filters.doc_type else None,
    }
    terms = {key: value for key, value in candidates.items() if value is not None}
    if filters.path_prefix:
        # `ancestors` holds every directory prefix, so a directory restriction
        # is an exact match against an array element rather than a prefix scan.
        # Neither engine can prefix-match a keyword efficiently.
        terms["ancestors"] = filters.path_prefix.strip("/")
    return terms


def vector_filter(filters: SearchFilters | None) -> dict[str, Any] | None:
    """The `filter` document for a `$vectorSearch` stage.

    MQL, but only the comparison subset Atlas allows there -- no `$regex`, no
    `$expr`. Everything we need is equality or set membership, so that costs us
    nothing.
    """
    if filters is None or filters.is_empty():
        return None

    clauses: list[dict[str, Any]] = [
        {key: {"$eq": value}} for key, value in exact_terms(filters).items()
    ]
    if filters.doc_types:
        # `$in`, not several `$eq` clauses under an `$and`: asking for a chunk
        # that is somehow both normative and design matches nothing at all.
        clauses.append({"doc_type": {"$in": [t.value for t in filters.doc_types]}})
    if filters.exclude_doc_types:
        # `$nin` rather than a positive list, so a type added to the taxonomy
        # later is included by default rather than silently dropped.
        clauses.append({"doc_type": {"$nin": [t.value for t in filters.exclude_doc_types]}})

    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def search_clauses(
    filters: SearchFilters | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The `filter` and `mustNot` clauses for a `$search` compound operator.

    Returned as `filter` rather than `must` on purpose: a `filter` clause
    constrains without contributing to the relevance score, so restricting to
    one repository cannot change the ranking within it. `must` would make every
    surviving document score higher for matching a constraint they all match.
    """
    if filters is None or filters.is_empty():
        return [], []

    include: list[dict[str, Any]] = [
        {"equals": {"path": key, "value": value}} for key, value in exact_terms(filters).items()
    ]
    if filters.doc_types:
        include.append({"in": {"path": "doc_type", "value": [t.value for t in filters.doc_types]}})
    exclude: list[dict[str, Any]] = [
        {"equals": {"path": "doc_type", "value": t.value}} for t in filters.exclude_doc_types
    ]
    return include, exclude


def match_stage(filters: SearchFilters | None) -> dict[str, Any]:
    """A plain MQL `$match` body, for the non-search paths.

    `count`, `facet` and `scroll` are ordinary aggregations with no relevance
    involved, so they use this rather than either search dialect. Same terms,
    so a count and a search agree about what the filter means -- if they
    diverged, `status` would report totals no search could reproduce.
    """
    if filters is None or filters.is_empty():
        return {}

    # Operator form throughout, so several constraints on the same field
    # combine instead of overwriting each other. A field expression holding
    # more than one operator is an implicit AND, which is exactly what
    # doc_type="normative" plus exclude_doc_types=["test"] has to mean; a bare
    # `{"doc_type": "normative"}` would be clobbered by the exclusion, or
    # clobber it, depending on assignment order.
    match: dict[str, dict[str, Any]] = {
        key: {"$eq": value} for key, value in exact_terms(filters).items()
    }
    if filters.doc_types:
        match.setdefault("doc_type", {})["$in"] = [t.value for t in filters.doc_types]
    if filters.exclude_doc_types:
        match.setdefault("doc_type", {})["$nin"] = [t.value for t in filters.exclude_doc_types]
    return dict(match)
