"""Real embeddings, no API key.

pydantic-ai's TestEmbeddingModel returns all-ones vectors, so it can verify
plumbing but never relevance: every pair is equally similar. This module runs a
genuine local model so "does retrieval actually work" is an assertion rather
than an assumption.

Marked `integration` because the first run downloads ~130 MB (cached after).
Still needs no API key, no credentials and no paid request — skip with
`-m "not integration"` for a fast offline loop.
"""

from __future__ import annotations

import math

import pytest

from workspace_indexer.embedding.embedding_service import EmbeddingService
from workspace_indexer.embedding.fastembed_dense_backend import FastembedDenseBackend

pytestmark = pytest.mark.integration

AUTH_DOC = (
    "Authentication is handled by verifying the bearer token on each request "
    "and rejecting expired credentials before the handler runs."
)
DESSERT_DOC = (
    "Cream the butter and sugar, fold in the flour, then bake the sponge cake "
    "for forty minutes until golden."
)
DEPLOY_DOC = (
    "To roll back a deployment, run the rollback script and page the on-call "
    "engineer if the health checks keep failing."
)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


@pytest.fixture(scope="module")
def backend() -> FastembedDenseBackend:
    return FastembedDenseBackend()


async def test_dimensions_match_the_declared_space(backend: FastembedDenseBackend) -> None:
    """A mismatch here would make the service's dimension guard fire on every
    batch."""
    vectors = await backend.embed_documents(["hello"])
    assert len(vectors[0]) == backend.space.dimensions


async def test_semantically_close_text_scores_higher(backend: FastembedDenseBackend) -> None:
    """The property the whole index rests on: meaning beats keyword overlap.
    The query shares no significant words with the auth document."""
    docs = await backend.embed_documents([AUTH_DOC, DESSERT_DOC])
    query = await backend.embed_query("how do we check a user is logged in")
    assert _cosine(query, docs[0]) > _cosine(query, docs[1])


async def test_ranks_the_right_document_first(backend: FastembedDenseBackend) -> None:
    docs = await backend.embed_documents([AUTH_DOC, DESSERT_DOC, DEPLOY_DOC])
    query = await backend.embed_query("how do I undo a bad release")
    scores = [_cosine(query, d) for d in docs]
    assert scores.index(max(scores)) == 2, scores


async def test_identical_text_is_near_identical_vector(
    backend: FastembedDenseBackend,
) -> None:
    """Determinism: re-embedding unchanged content must not churn the index."""
    first, second = await backend.embed_documents([AUTH_DOC, AUTH_DOC])
    assert _cosine(first, second) > 0.999


async def test_service_batches_a_real_backend_in_order(
    backend: FastembedDenseBackend,
) -> None:
    """Ordering across concurrent batches, verified against a real model rather
    than a fake that could be ordered by construction."""
    docs = [AUTH_DOC, DESSERT_DOC, DEPLOY_DOC]
    service = EmbeddingService(backend, batch_size=1, max_concurrency=3)
    batched = await service.embed_documents(docs)
    direct = await backend.embed_documents(docs)
    for got, want in zip(batched, direct, strict=True):
        assert _cosine(got, want) > 0.999
    assert service.stats.documents == 3
    assert service.stats.requests == 3


async def test_local_inference_reports_zero_cost_not_unknown(
    backend: FastembedDenseBackend,
) -> None:
    service = EmbeddingService(backend)
    await service.embed_documents(["anything"])
    assert service.stats.est_cost_usd == 0.0
    assert service.stats.unpriced_requests == 0
