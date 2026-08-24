"""The read path, end to end against a real embedded Qdrant."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from tests.conftest import make_source
from tests.fake_embedding_backend import FakeEmbeddingBackend
from tests.fake_scoring_reranker import FakeScoringReranker
from tests.fake_sparse_backend import FakeSparseBackend
from workspace_indexer.chunking.chunk_factory import build_chunk
from workspace_indexer.config import RerankConfig, SearchSection
from workspace_indexer.embedding.embedding_service import EmbeddingService
from workspace_indexer.models import Chunk, EmbeddingSpace, FileKind, SearchFilters
from workspace_indexer.rerank.noop_reranker import NoopReranker
from workspace_indexer.rerank.reranker import Reranker
from workspace_indexer.search.search_request import SearchRequest
from workspace_indexer.search.search_service import SearchService
from workspace_indexer.storage.qdrant_store import QdrantStore

SPACE = EmbeddingSpace(model="fake:model", dimensions=4)

DOCS = [
    ("src/auth/login.py", "def login(): return check_token()"),
    ("src/bake/cake.py", "def bake(): return sponge()"),
    ("docs/deploy.md", "rollback the release and page on-call"),
    ("src/util/misc.py", "def helper(): return 1"),
]


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[QdrantStore]:
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace="labbox", payload_indexes=False)
    sparse = FakeSparseBackend()
    chunks: list[Chunk] = []
    for index, (rel_path, body) in enumerate(DOCS):
        source = make_source(
            body,
            kind=FileKind.CODE,
            language="python",
            rel_path=rel_path,
            unit="repo_one" if rel_path.startswith("src") else "docs",
        )
        chunks.append(
            build_chunk(
                source,
                "labbox",
                source_text=body,
                start_line=1,
                end_line=1,
                chunker="code",
                version=1,
                chunk_index=index,
                symbol_path=f"sym{index}",
            )
        )
    dense = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]
    await store.upsert(SPACE, chunks, dense, sparse.encode_documents([b for _, b in DOCS]))
    yield store
    await client.close()


def _service(
    store: QdrantStore,
    *,
    reranker: Reranker | None = None,
    config: SearchSection | None = None,
) -> SearchService:
    backend = FakeEmbeddingBackend(dimensions=4)
    return SearchService(
        store=store,
        embeddings=EmbeddingService(backend),
        sparse=FakeSparseBackend(),
        reranker=reranker or NoopReranker(),
        config=config or SearchSection(rerank=RerankConfig(enabled=False, model="fake:m")),
        space=SPACE,
    )


async def test_returns_hits(store: QdrantStore) -> None:
    hits = await _service(store).search(SearchRequest(query="login"))
    assert hits
    assert all(h.rel_path for h in hits)


async def test_limit_is_honoured(store: QdrantStore) -> None:
    hits = await _service(store).search(SearchRequest(query="login", limit=2))
    assert len(hits) == 2


async def test_default_limit_comes_from_config(store: QdrantStore) -> None:
    config = SearchSection(default_limit=3, rerank=RerankConfig(enabled=False, model="fake:m"))
    hits = await _service(store, config=config).search(SearchRequest(query="login"))
    assert len(hits) == 3


async def test_filters_are_passed_through(store: QdrantStore) -> None:
    hits = await _service(store).search(
        SearchRequest(query="anything", filters=SearchFilters(unit="docs"))
    )
    assert [h.rel_path for h in hits] == ["docs/deploy.md"]


async def test_dense_only_skips_the_sparse_encoder(store: QdrantStore) -> None:
    """Only pay for the branch the fusion mode will use."""
    sparse = FakeSparseBackend()
    service = SearchService(
        store=store,
        embeddings=EmbeddingService(FakeEmbeddingBackend(dimensions=4)),
        sparse=sparse,
        reranker=NoopReranker(),
        config=SearchSection(rerank=RerankConfig(enabled=False, model="fake:m")),
        space=SPACE,
    )
    await service.search(SearchRequest(query="login", fusion="dense_only"))
    assert sparse.queries == []


async def test_sparse_only_skips_the_dense_embedding(store: QdrantStore) -> None:
    backend = FakeEmbeddingBackend(dimensions=4)
    service = SearchService(
        store=store,
        embeddings=EmbeddingService(backend),
        sparse=FakeSparseBackend(),
        reranker=NoopReranker(),
        config=SearchSection(rerank=RerankConfig(enabled=False, model="fake:m")),
        space=SPACE,
    )
    await service.search(SearchRequest(query="rollback", fusion="sparse_only"))
    assert backend.queries == []


async def test_request_fusion_overrides_the_config(store: QdrantStore) -> None:
    """`--fusion dense` is a debugging tool: when a query returns junk, the
    first question is which branch produced it."""
    sparse = FakeSparseBackend()
    service = SearchService(
        store=store,
        embeddings=EmbeddingService(FakeEmbeddingBackend(dimensions=4)),
        sparse=sparse,
        reranker=NoopReranker(),
        config=SearchSection(fusion="rrf", rerank=RerankConfig(enabled=False, model="fake:m")),
        space=SPACE,
    )
    await service.search(SearchRequest(query="login", fusion="dense_only"))
    assert sparse.queries == []


async def test_reranker_reorders_the_results(store: QdrantStore) -> None:
    reranker = FakeScoringReranker(RerankConfig(model="fake:m"), reverse=True)
    config = SearchSection(rerank=RerankConfig(model="fake:m", candidates=4, top_n=4))
    hits = await _service(store, reranker=reranker, config=config).search(
        SearchRequest(query="login", limit=4)
    )
    assert reranker.stats.calls == 1
    assert len(hits) == 4


async def test_retrieval_goes_deep_and_returns_shallow(store: QdrantStore) -> None:
    """The reranker only needs the right chunk somewhere in the candidate set;
    it decides the final order. So we fetch `candidates`, not `limit`."""
    reranker = FakeScoringReranker(RerankConfig(model="fake:m"))
    config = SearchSection(rerank=RerankConfig(model="fake:m", candidates=4))
    hits = await _service(store, reranker=reranker, config=config).search(
        SearchRequest(query="login", limit=1)
    )
    assert len(reranker.seen_documents[0]) == 4
    assert len(hits) == 1


async def test_rerank_can_be_turned_off_per_request(store: QdrantStore) -> None:
    """A per-call override swaps the object rather than setting a flag."""
    reranker = FakeScoringReranker(RerankConfig(model="fake:m"), reverse=True)
    config = SearchSection(rerank=RerankConfig(model="fake:m", candidates=4))
    hits = await _service(store, reranker=reranker, config=config).search(
        SearchRequest(query="login", limit=4, rerank=False)
    )
    assert reranker.stats.calls == 0
    assert len(hits) == 4


async def test_a_failing_reranker_still_returns_results(store: QdrantStore) -> None:
    """Results get worse; the search does not fail."""
    reranker = FakeScoringReranker(RerankConfig(model="fake:m"), error=RuntimeError("503"))
    config = SearchSection(rerank=RerankConfig(model="fake:m", candidates=4))
    hits = await _service(store, reranker=reranker, config=config).search(
        SearchRequest(query="login", limit=3)
    )
    assert len(hits) == 3
    assert reranker.stats.degraded == 1


async def test_staleness_is_checked_by_default(store: QdrantStore) -> None:
    """The fixture's files do not exist on disk, so every hit is stale."""
    hits = await _service(store).search(SearchRequest(query="login", limit=2))
    assert all(h.stale for h in hits)


async def test_staleness_can_be_skipped(store: QdrantStore) -> None:
    """The eval harness does not need the I/O."""
    hits = await _service(store).search(
        SearchRequest(query="login", limit=2, check_staleness=False)
    )
    assert not any(h.stale for h in hits)


async def test_fresh_files_are_not_flagged(tmp_path: Path) -> None:
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace="labbox", payload_indexes=False)
    body = "def login(): return check_token()"
    real = tmp_path / "login.py"
    real.write_text(body, encoding="utf-8")

    source = make_source(body, kind=FileKind.CODE, language="python", rel_path="login.py")
    chunk = build_chunk(
        source.model_copy(update={"abs_path": real}),
        "labbox",
        source_text=body,
        start_line=1,
        end_line=1,
        chunker="code",
        version=1,
    )
    sparse = FakeSparseBackend()
    await store.upsert(SPACE, [chunk], [[1.0, 0, 0, 0]], sparse.encode_documents([body]))

    hits = await _service(store).search(SearchRequest(query="login"))
    assert hits and not hits[0].stale
    await client.close()


async def test_empty_collection_returns_nothing(tmp_path: Path) -> None:
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace="labbox", payload_indexes=False)
    await store.ensure_collection(SPACE)
    assert await _service(store).search(SearchRequest(query="anything")) == []
    await client.close()
