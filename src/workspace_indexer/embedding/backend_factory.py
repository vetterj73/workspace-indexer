"""Build the embedding stack from settings.

One place that knows how a config string becomes a backend, so the pipeline and
the CLI never branch on provider names.
"""

from __future__ import annotations

from workspace_indexer.config import Settings
from workspace_indexer.embedding.embedding_backend import EmbeddingBackend
from workspace_indexer.embedding.embedding_service import EmbeddingService
from workspace_indexer.embedding.fastembed_dense_backend import FastembedDenseBackend
from workspace_indexer.embedding.fastembed_sparse_backend import FastembedSparseBackend
from workspace_indexer.embedding.pydantic_ai_backend import PydanticAiBackend
from workspace_indexer.embedding.sparse_backend import SparseBackend
from workspace_indexer.embedding.token_pricer import TokenPricer
from workspace_indexer.models import EmbeddingSpace
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.embedding.factory")

# Models served locally by fastembed rather than through pydantic-ai. Spelled as
# a prefix so the rest of the system keeps one `provider:model` convention.
_LOCAL_PREFIX = "fastembed:"


def build_space(settings: Settings) -> EmbeddingSpace:
    return EmbeddingSpace(
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        sparse_model=settings.sparse_model,
    )


def build_dense_backend(settings: Settings) -> EmbeddingBackend:
    model = settings.embedding_model
    if model.startswith(_LOCAL_PREFIX):
        name = model[len(_LOCAL_PREFIX) :]
        log.info("embed.backend", kind="fastembed", model=name)
        return FastembedDenseBackend(name, dimensions=settings.embedding_dimensions)
    exported = settings.export_credentials()
    log.info("embed.backend", kind="pydantic-ai", model=model, credentials=exported)
    return PydanticAiBackend(build_space(settings))


def build_sparse_backend(settings: Settings) -> SparseBackend:
    return FastembedSparseBackend(settings.sparse_model)


def build_embedding_service(settings: Settings) -> EmbeddingService:
    return EmbeddingService(
        build_dense_backend(settings),
        batch_size=settings.embedding_batch_size,
        max_concurrency=settings.embedding_max_concurrency,
        max_batch_tokens=settings.embedding_max_batch_tokens,
        pricer=TokenPricer(settings.embedding_price_per_mtok),
    )
