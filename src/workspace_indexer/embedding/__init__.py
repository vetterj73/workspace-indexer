"""Embedding: dense via a swappable provider, sparse BM25 locally."""

from workspace_indexer.embedding.backend_factory import (
    build_dense_backend,
    build_embedding_service,
    build_space,
    build_sparse_backend,
)
from workspace_indexer.embedding.embedding_backend import EmbeddingBackend
from workspace_indexer.embedding.embedding_service import EmbeddingService
from workspace_indexer.embedding.embedding_stats import EmbeddingStats
from workspace_indexer.embedding.fastembed_dense_backend import FastembedDenseBackend
from workspace_indexer.embedding.fastembed_sparse_backend import FastembedSparseBackend
from workspace_indexer.embedding.pydantic_ai_backend import PydanticAiBackend
from workspace_indexer.embedding.retry_policy import RetryPolicy
from workspace_indexer.embedding.sparse_backend import SparseBackend

__all__ = [
    "EmbeddingBackend",
    "EmbeddingService",
    "EmbeddingStats",
    "FastembedDenseBackend",
    "FastembedSparseBackend",
    "PydanticAiBackend",
    "RetryPolicy",
    "SparseBackend",
    "build_dense_backend",
    "build_embedding_service",
    "build_sparse_backend",
    "build_space",
]
