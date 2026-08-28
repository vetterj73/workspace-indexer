"""The two Atlas indexes a collection needs, and the fields they must declare.

Two indexes because hybrid search needs two engines: `$vectorSearch` over the
dense vector and `$search` over the text. Both are mongot indexes, built
asynchronously, and both count against the cluster's index budget -- a Free
cluster allows three of either kind in total, so this uses two of three and
leaves one spare.

The single most important rule here is that FILTER_FIELDS is one list used by
both index definitions and by `mongo_filter`. Atlas silently refuses to filter
on a field an index did not declare -- `$vectorSearch` errors, `$search`
`equals` fails -- so a field that drifts out of one of the three places is a
filter that stops working, and the symptom is a search that returns nothing
rather than a message saying why.
"""

from __future__ import annotations

from typing import Any

VECTOR_INDEX = "dense_vector"
TEXT_INDEX = "source_text"

# The field holding the dense embedding. Named to match Qdrant's named vector
# so the payload contract reads the same on both backends.
DENSE_FIELD = "dense"

# What both engines are allowed to filter on. Mirrors Qdrant's INDEXED_FIELDS:
# the payload is a contract shared by every backend, and a filter that works on
# one store and not the other would make the choice of store visible to the MCP
# layer, which is the whole thing the VectorStore protocol exists to prevent.
FILTER_FIELDS = (
    "workspace",
    "root_label",
    "unit",
    "rel_path",
    "ancestors",
    "ext",
    "kind",
    "language",
    "repo_name",
    "repo_branch",
    "symbol_kind",
    "doc_type",
    "content_sha",
    "space_slug",
)

# What `$search` scores against. Both halves, because Qdrant's BM25 branch
# encodes `embed_text` -- the context header prepended to the source -- and we
# store those as two fields rather than duplicating the source text. Indexing
# both paths gives Lucene the same text to work with, so the sparse half of
# hybrid search means the same thing on both backends.
TEXT_FIELDS = ("source_text", "context_header", "symbol_path")

# Atlas's own names for the vector element type. `float32` is the default and
# the safe choice; `int8` is a 4x storage cut that costs some recall, and is
# worth reaching for only when a Free cluster's 512 MB is the binding
# constraint. Deliberately not `int1`: it supports euclidean similarity only,
# and every measurement this project has taken is on cosine.
DTYPES = ("float32", "int8")


def vector_index(dimensions: int, *, dtype: str = "float32") -> dict[str, Any]:
    """The `$vectorSearch` index definition.

    Cosine, matching Qdrant, so the eval numbers from one backend mean
    something on the other. `filter` entries are declared for every field in
    FILTER_FIELDS whether or not a given query uses them -- declaring one costs
    almost nothing, and discovering a missing one costs a failed search in
    production.
    """
    if dtype not in DTYPES:
        raise ValueError(f"unsupported vector dtype {dtype!r}; expected one of {DTYPES}")
    fields: list[dict[str, Any]] = [
        {
            "type": "vector",
            "path": DENSE_FIELD,
            "numDimensions": dimensions,
            "similarity": "cosine",
        }
    ]
    fields.extend({"type": "filter", "path": field} for field in FILTER_FIELDS)
    return {"fields": fields}


def text_index() -> dict[str, Any]:
    """The `$search` index definition: BM25 over the text, tokens for filters.

    `dynamic: False` with fields named explicitly. A dynamic index would map
    every field including the vector, which on a Free cluster is a large amount
    of index built to answer questions nobody asks.

    Filter fields are mapped as `token`, not `string`. A `string` mapping is
    analysed -- lowercased and split -- so an `equals` clause against a path
    like `src/workspace_indexer` would match nothing at all, and the search
    would come back empty with no error to explain it.
    """
    mappings: dict[str, Any] = {field: {"type": "string"} for field in TEXT_FIELDS}
    mappings.update({field: {"type": "token"} for field in FILTER_FIELDS})
    return {"mappings": {"dynamic": False, "fields": mappings}}
