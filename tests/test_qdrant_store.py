"""The Qdrant store, against a real embedded instance.

Embedded mode is a real Qdrant, not a mock: it supports named vectors, sparse
vectors, RRF fusion and server-side filters, all verified here. So these run by
default with no server, no network and no API key.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from tests.conftest import make_source
from workspace_indexer.chunking.chunk_factory import build_chunk
from workspace_indexer.models import (
    Chunk,
    EmbeddingSpace,
    FileKind,
    SearchFilters,
    SparseVec,
)
from workspace_indexer.storage.qdrant_store import DENSE_VECTOR, SPARSE_VECTOR, QdrantStore
from workspace_indexer.storage.query_spec import QuerySpec

SPACE = EmbeddingSpace(model="fake:model", dimensions=4)
OTHER_SPACE = EmbeddingSpace(model="fake:model", dimensions=8)


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[QdrantStore]:
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    # Embedded Qdrant ignores payload indexes and warns once per field.
    yield QdrantStore(client, workspace="labbox", payload_indexes=False)
    await client.close()


def _chunk(
    rel_path: str,
    text: str,
    *,
    unit: str = "repo_one",
    language: str = "python",
    kind: FileKind = FileKind.CODE,
) -> Chunk:
    file = make_source(text, kind=kind, language=language, rel_path=rel_path, unit=unit)
    return build_chunk(
        file,
        "labbox",
        source_text=text,
        start_line=1,
        end_line=max(1, len(text.splitlines())),
        chunker="code",
        version=1,
        symbol_path="Thing.method",
        symbol_kind="function",
        symbol_name="method",
    )


def _dense(*values: float) -> list[float]:
    return list(values)


def _sparse(*indices: int) -> SparseVec:
    return SparseVec(indices=list(indices), values=[1.0] * len(indices))


async def _seed(store: QdrantStore) -> list[Chunk]:
    chunks = [
        _chunk("src/auth/login.py", "def login(): pass"),
        _chunk("src/bake/cake.py", "def bake(): pass"),
        _chunk(
            "docs/deploy.md",
            "rollback steps",
            unit="plain_folder",
            language="markdown",
            kind=FileKind.MARKDOWN,
        ),
    ]
    dense = [_dense(1, 0, 0, 0), _dense(0, 1, 0, 0), _dense(0, 0, 1, 0)]
    sparse = [_sparse(10, 11), _sparse(20, 21), _sparse(30, 31)]
    await store.upsert(SPACE, chunks, dense, sparse)
    return chunks


def test_collection_name_includes_workspace_and_space(store: QdrantStore) -> None:
    """One collection per embedding space, because vectors from two models are
    not comparable."""
    assert store.collection_name(SPACE) == "labbox__fake_model_4"
    assert store.collection_name(OTHER_SPACE) == "labbox__fake_model_8"


async def test_ensure_collection_declares_both_named_vectors(store: QdrantStore) -> None:
    """Adding a named vector to a populated collection is not a simple
    migration, so the sparse half has to exist from the first run."""
    await store.ensure_collection(SPACE)
    declared = await store.describe_vectors(SPACE)
    assert declared == {"dense": [DENSE_VECTOR], "sparse": [SPARSE_VECTOR]}


async def test_ensure_collection_is_idempotent(store: QdrantStore) -> None:
    await store.ensure_collection(SPACE)
    await _seed(store)
    await store.ensure_collection(SPACE)
    assert await store.count(SPACE) == 3


async def test_upsert_then_count(store: QdrantStore) -> None:
    await _seed(store)
    assert await store.count(SPACE) == 3


async def test_count_of_a_missing_collection_is_zero_not_an_error(store: QdrantStore) -> None:
    assert await store.count(OTHER_SPACE) == 0


async def test_upsert_creates_the_collection_on_demand(store: QdrantStore) -> None:
    await _seed(store)
    assert store.collection_name(SPACE) in await store.collection_names()


async def test_empty_upsert_is_a_no_op(store: QdrantStore) -> None:
    await store.upsert(SPACE, [], [], [])
    assert await store.count(SPACE) == 0


async def test_size_mismatch_is_rejected(store: QdrantStore) -> None:
    """A mismatch would attach one chunk's vector to another's payload, and
    nothing downstream could detect it."""
    chunk = _chunk("a.py", "x = 1")
    with pytest.raises(ValueError, match="size mismatch"):
        await store.upsert(SPACE, [chunk], [_dense(1, 0, 0, 0), _dense(0, 1, 0, 0)], [_sparse(1)])


async def test_reupserting_the_same_id_replaces_rather_than_duplicates(
    store: QdrantStore,
) -> None:
    """Content-addressed ids plus upsert is what makes reindexing idempotent."""
    chunk = _chunk("a.py", "x = 1")
    for _ in range(3):
        await store.upsert(SPACE, [chunk], [_dense(1, 0, 0, 0)], [_sparse(1)])
    assert await store.count(SPACE) == 1


async def test_dense_search_ranks_by_vector_similarity(store: QdrantStore) -> None:
    await _seed(store)
    hits = await store.search(
        SPACE, QuerySpec(dense=_dense(1, 0, 0, 0), fusion="dense_only", limit=3)
    )
    assert hits[0].rel_path == "src/auth/login.py"


async def test_sparse_search_ranks_by_term_overlap(store: QdrantStore) -> None:
    await _seed(store)
    hits = await store.search(
        SPACE, QuerySpec(sparse=_sparse(30, 31), fusion="sparse_only", limit=3)
    )
    assert hits[0].rel_path == "docs/deploy.md"


async def test_rrf_surfaces_both_branches_winners(store: QdrantStore) -> None:
    """The test that proves hybrid earns its complexity: the dense branch
    favours one document and the sparse branch another, and fusion has to
    return both rather than letting one branch dominate."""
    await _seed(store)
    hits = await store.search(
        SPACE,
        QuerySpec(dense=_dense(1, 0, 0, 0), sparse=_sparse(30, 31), fusion="rrf", limit=3),
    )
    paths = [h.rel_path for h in hits[:2]]
    assert "src/auth/login.py" in paths
    assert "docs/deploy.md" in paths


async def test_missing_branch_degrades_instead_of_raising(store: QdrantStore) -> None:
    await _seed(store)
    assert await store.search(SPACE, QuerySpec(fusion="rrf")) == []
    assert await store.search(SPACE, QuerySpec(fusion="dense_only")) == []
    assert await store.search(SPACE, QuerySpec(sparse=None, fusion="sparse_only")) == []


async def test_rrf_works_with_only_the_dense_branch_present(store: QdrantStore) -> None:
    await _seed(store)
    hits = await store.search(SPACE, QuerySpec(dense=_dense(1, 0, 0, 0), fusion="rrf", limit=2))
    assert hits


async def test_filters_are_applied_inside_the_search(store: QdrantStore) -> None:
    """Not after it: post-filtering a page would return 3 hits when 10 were
    asked for, because the other 7 were elsewhere."""
    await _seed(store)
    hits = await store.search(
        SPACE,
        QuerySpec(dense=_dense(1, 0, 0, 0), fusion="dense_only", limit=10),
        SearchFilters(unit="plain_folder"),
    )
    assert [h.rel_path for h in hits] == ["docs/deploy.md"]


async def test_path_prefix_filter_uses_ancestors(store: QdrantStore) -> None:
    await _seed(store)
    hits = await store.search(
        SPACE,
        QuerySpec(dense=_dense(1, 0, 0, 0), fusion="rrf", sparse=_sparse(10), limit=10),
        SearchFilters(path_prefix="src/auth"),
    )
    assert [h.rel_path for h in hits] == ["src/auth/login.py"]


async def test_kind_filter(store: QdrantStore) -> None:
    await _seed(store)
    hits = await store.search(
        SPACE,
        QuerySpec(dense=_dense(0, 0, 1, 0), fusion="dense_only", limit=10),
        SearchFilters(kind=FileKind.MARKDOWN),
    )
    assert [h.rel_path for h in hits] == ["docs/deploy.md"]


async def test_hits_carry_everything_a_result_needs(store: QdrantStore) -> None:
    chunks = await _seed(store)
    hits = await store.search(
        SPACE, QuerySpec(dense=_dense(1, 0, 0, 0), fusion="dense_only", limit=1)
    )
    hit = hits[0]
    assert hit.location == "src/auth/login.py:1-1"
    assert hit.source_text == "def login(): pass"
    assert hit.symbol_path == "Thing.method"
    assert hit.content_sha == chunks[0].meta.content_sha
    assert hit.embed_text == chunks[0].embed_text


async def test_delete_by_ids(store: QdrantStore) -> None:
    chunks = await _seed(store)
    await store.delete_by_ids(SPACE, [chunks[0].chunk_id])
    assert await store.count(SPACE) == 2


async def test_empty_delete_is_a_no_op(store: QdrantStore) -> None:
    await _seed(store)
    await store.delete_by_ids(SPACE, [])
    assert await store.count(SPACE) == 3


async def test_delete_by_path_removes_every_chunk_of_a_file(store: QdrantStore) -> None:
    """What makes deletion and rename correct without consulting the manifest
    first: one call clears a file whatever it chunked into."""
    parts = [
        _chunk("src/big.py", "part one body"),
        _chunk("src/big.py", "part two body"),
    ]
    await store.upsert(SPACE, parts, [_dense(1, 0, 0, 0), _dense(0, 1, 0, 0)],
                       [_sparse(1), _sparse(2)])
    assert await store.count(SPACE) == 2
    await store.delete_by_path(SPACE, "repo_one", "src/big.py")
    assert await store.count(SPACE) == 0


async def test_delete_by_path_is_scoped_to_the_root(store: QdrantStore) -> None:
    """The same rel_path can exist under two roots; deleting one must not
    touch the other."""
    await _seed(store)
    await store.delete_by_path(SPACE, "other_root", "src/auth/login.py")
    assert await store.count(SPACE) == 3


async def test_scroll_yields_payloads_and_named_vectors(store: QdrantStore) -> None:
    """This is what lets reproject build a truncated collection from vectors we
    already paid for."""
    await _seed(store)
    seen: list[tuple[str, str, list[str]]] = []
    async for point_id, payload, vectors in store.scroll(SPACE, with_vectors=True):
        assert vectors is not None
        seen.append((point_id, str(payload["rel_path"]), sorted(vectors)))
    assert len(seen) == 3
    assert all(v == [SPARSE_VECTOR, DENSE_VECTOR] or v == sorted([DENSE_VECTOR, SPARSE_VECTOR])
               for _, _, v in seen)


async def test_scroll_without_vectors_reports_none(store: QdrantStore) -> None:
    await _seed(store)
    async for _, _, vectors in store.scroll(SPACE, with_vectors=False):
        assert vectors is None


async def test_scroll_pages_through_everything(store: QdrantStore) -> None:
    chunks = [_chunk(f"src/f{i}.py", f"body number {i}") for i in range(25)]
    await store.upsert(
        SPACE,
        chunks,
        [_dense(float(i), 0, 0, 0) for i in range(25)],
        [_sparse(i) for i in range(25)],
    )
    ids = [pid async for pid, _, _ in store.scroll(SPACE, batch_size=4)]
    assert len(ids) == 25
    assert len(set(ids)) == 25


async def test_spaces_are_isolated_from_each_other(store: QdrantStore) -> None:
    """The model-swap path: a new space backfills a new collection without
    disturbing the old one."""
    await _seed(store)
    chunk = _chunk("src/auth/login.py", "def login(): pass")
    await store.upsert(OTHER_SPACE, [chunk], [[0.0] * 8], [_sparse(1)])
    assert await store.count(SPACE) == 3
    assert await store.count(OTHER_SPACE) == 1
    assert len(await store.collection_names()) == 2


async def test_drop_collection(store: QdrantStore) -> None:
    await _seed(store)
    await store.drop_collection(SPACE)
    assert store.collection_name(SPACE) not in await store.collection_names()
    # And can be recreated afterwards.
    await _seed(store)
    assert await store.count(SPACE) == 3


async def test_dropping_a_missing_collection_does_not_raise(store: QdrantStore) -> None:
    await store.drop_collection(OTHER_SPACE)
