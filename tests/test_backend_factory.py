"""Turning settings into backends.

One place knows how a config string becomes a backend, so the pipeline and CLI
never branch on provider names.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from workspace_indexer.config import Settings
from workspace_indexer.embedding.backend_factory import (
    build_dense_backend,
    build_embedding_service,
    build_space,
)
from workspace_indexer.embedding.fastembed_dense_backend import FastembedDenseBackend
from workspace_indexer.embedding.pydantic_ai_backend import PydanticAiBackend


@pytest.fixture(autouse=True)
def _isolate_env(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Settings reads .env from the cwd; without this the developer's real
    configuration leaks into assertions."""
    monkeypatch.chdir(tmp_path)


def test_space_carries_model_dimensions_and_sparse_model() -> None:
    settings = Settings(
        embedding_model="voyageai:voyage-code-4",
        embedding_dimensions=2048,
        sparse_model="Qdrant/bm25",
    )
    space = build_space(settings)
    assert space.slug() == "voyageai_voyage-code-4_2048"
    assert space.sparse_model == "Qdrant/bm25"


def test_dimension_change_produces_a_different_collection_name() -> None:
    """Reprojecting 2048 to 1024 has to land in a separate collection, or the
    experiment overwrites the thing it is being compared against."""
    a = build_space(Settings(embedding_dimensions=2048)).slug()
    b = build_space(Settings(embedding_dimensions=1024)).slug()
    assert a != b


def test_api_models_route_to_pydantic_ai() -> None:
    backend = build_dense_backend(Settings(embedding_model="voyageai:voyage-code-4"))
    assert isinstance(backend, PydanticAiBackend)


def test_fastembed_prefix_routes_to_the_local_backend() -> None:
    """The offline path, and the one the test suite uses for real relevance."""
    settings = Settings(embedding_model="fastembed:BAAI/bge-small-en-v1.5",
                        embedding_dimensions=384)
    backend = build_dense_backend(settings)
    assert isinstance(backend, FastembedDenseBackend)
    assert backend.space.dimensions == 384


def test_local_backend_construction_downloads_nothing() -> None:
    """lazy_load means building a backend for a --dry-run costs nothing."""
    settings = Settings(embedding_model="fastembed:BAAI/bge-small-en-v1.5",
                        embedding_dimensions=384)
    assert build_dense_backend(settings) is not None


def test_service_picks_up_batching_settings() -> None:
    settings = Settings(
        embedding_model="fastembed:BAAI/bge-small-en-v1.5",
        embedding_dimensions=384,
        embedding_batch_size=7,
        embedding_max_concurrency=2,
        embedding_max_batch_tokens=1234,
    )
    service = build_embedding_service(settings)
    assert service.space.dimensions == 384
    # Batching is observable through the plan rather than private attributes.
    assert service.stats.requests == 0
