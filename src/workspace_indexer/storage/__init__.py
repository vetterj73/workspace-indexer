"""Storage: a VectorStore protocol over Qdrant, one collection per space."""

from workspace_indexer.storage.payload import (
    INDEXED_FIELDS,
    ancestors_of,
    to_payload,
    to_search_hit,
)
from workspace_indexer.storage.qdrant_store import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    QdrantStore,
    build_filter,
)
from workspace_indexer.storage.query_spec import QuerySpec
from workspace_indexer.storage.store_factory import build_qdrant_client, build_vector_store
from workspace_indexer.storage.vector_store import VectorStore

__all__ = [
    "DENSE_VECTOR",
    "INDEXED_FIELDS",
    "SPARSE_VECTOR",
    "QdrantStore",
    "QuerySpec",
    "VectorStore",
    "ancestors_of",
    "build_filter",
    "build_qdrant_client",
    "build_vector_store",
    "to_payload",
    "to_search_hit",
]
