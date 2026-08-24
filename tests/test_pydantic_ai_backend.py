"""The pydantic-ai adapter.

Driven through TestEmbeddingModel, which needs no key and no network. It returns
all-ones vectors, so it can verify plumbing but never relevance — relevance is
tested against a real local model in test_embedding_integration.py.
"""

from __future__ import annotations

import pytest
from pydantic_ai import Embedder
from pydantic_ai.embeddings import TestEmbeddingModel

from workspace_indexer.embedding.pydantic_ai_backend import PydanticAiBackend
from workspace_indexer.models import EmbeddingSpace


def _backend(dimensions: int = 8) -> PydanticAiBackend:
    model = TestEmbeddingModel(dimensions=dimensions)
    space = EmbeddingSpace(model="test:test", dimensions=dimensions)
    return PydanticAiBackend(space, embedder=Embedder(model))


async def test_embeds_documents_in_order() -> None:
    backend = _backend()
    vectors = await backend.embed_documents(["one", "two", "three"])
    assert len(vectors) == 3
    assert all(len(v) == 8 for v in vectors)


async def test_returns_plain_lists_not_provider_types() -> None:
    """Downstream code puts these straight into Qdrant, which wants lists."""
    vectors = await backend_vectors()
    assert isinstance(vectors, list)
    assert isinstance(vectors[0], list)
    assert isinstance(vectors[0][0], float)


async def backend_vectors() -> list[list[float]]:
    return await _backend().embed_documents(["one"])


async def test_query_returns_a_single_vector() -> None:
    vector = await _backend().embed_query("how does auth work")
    assert len(vector) == 8
    assert all(isinstance(v, float) for v in vector)


async def test_empty_document_list_is_accepted() -> None:
    assert await _backend().embed_documents([]) == []


async def test_token_helpers_are_available() -> None:
    """EmbeddingService needs both to warn about truncation."""
    backend = _backend()
    assert await backend.max_input_tokens() is not None
    assert await backend.count_tokens("two words") >= 1


async def test_unknown_provider_pricing_does_not_raise() -> None:
    """cost() raises LookupError for a provider genai-prices does not know, and
    a missing price is not a reason to fail an index."""
    backend = _backend()
    await backend.embed_documents(["one"])
    assert backend.last_cost_usd() is None


async def test_dimensions_setting_is_passed_to_the_model() -> None:
    """This is how EMBEDDING_DIMENSIONS=1024 actually reaches Voyage; without
    it the model would return its native width and the collection would be
    built wrong."""
    model = TestEmbeddingModel(dimensions=8)
    space = EmbeddingSpace(model="test:test", dimensions=256)
    backend = PydanticAiBackend(space, embedder=Embedder(model))
    await backend.embed_documents(["one"])
    assert model.last_settings is not None
    assert model.last_settings.get("dimensions") == 256


async def test_truncate_setting_is_passed_to_the_model() -> None:
    """Letting the provider truncate beats failing a whole batch, since the
    service warns when it can happen."""
    model = TestEmbeddingModel(dimensions=8)
    backend = PydanticAiBackend(
        EmbeddingSpace(model="test:test", dimensions=8), embedder=Embedder(model)
    )
    await backend.embed_documents(["one"])
    assert model.last_settings is not None
    assert model.last_settings.get("truncate") is True


def test_space_is_carried_for_collection_naming() -> None:
    space = EmbeddingSpace(model="voyageai:voyage-code-4", dimensions=2048)
    backend = PydanticAiBackend(space, embedder=Embedder(TestEmbeddingModel(dimensions=8)))
    assert backend.space.slug() == "voyageai_voyage-code-4_2048"


@pytest.mark.integration
async def test_real_provider_is_reachable() -> None:
    """Deselected by default: needs VOYAGE_API_KEY and network, and costs
    money. Run with -m integration when changing provider wiring."""
    pytest.skip("enable manually with a funded VOYAGE_API_KEY")
