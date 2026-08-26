"""Matryoshka reprojection.

The 2048-versus-1024 question is answerable by measurement rather than
guesswork precisely because this is free: the first 1024 entries of a 2048-d
voyage vector are themselves a valid 1024-d embedding. These tests prove the
free-experiment path works before any decision rests on it.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from tests.conftest import make_source
from tests.fake_sparse_backend import FakeSparseBackend
from workspace_indexer.chunking.chunk_factory import build_chunk
from workspace_indexer.models import Chunk, EmbeddingSpace, FileKind, SourceFile
from workspace_indexer.search.reprojector import Reprojector, truncate
from workspace_indexer.state import Manifest
from workspace_indexer.storage.qdrant_store import QdrantStore

WIDE = EmbeddingSpace(model="voyageai:voyage-code-4", dimensions=8)


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[QdrantStore]:
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    yield QdrantStore(client, workspace="labbox", payload_indexes=False)
    await client.close()


@pytest.fixture
def manifest(tmp_path: Path) -> Iterator[Manifest]:
    with Manifest(tmp_path / "manifest.sqlite3") as m:
        yield m


def _reprojector(store: QdrantStore, manifest: Manifest) -> Reprojector:
    return Reprojector(store, manifest)


async def _seed(store: QdrantStore, count: int = 5, manifest: Manifest | None = None) -> None:
    sparse = FakeSparseBackend()
    chunks: list[Chunk] = []
    vectors: list[list[float]] = []
    sources: list[SourceFile] = []
    for i in range(count):
        body = f"def function_{i}(): return {i}"
        source = make_source(body, kind=FileKind.CODE, language="python", rel_path=f"f{i}.py")
        sources.append(source)
        chunks.append(
            build_chunk(
                source,
                "labbox",
                source_text=body,
                start_line=1,
                end_line=1,
                chunker="code",
                version=1,
                symbol_path=f"function_{i}",
            )
        )
        vectors.append([float(i + 1)] * 8)
    await store.upsert(
        WIDE, chunks, vectors, sparse.encode_documents([c.source_text for c in chunks])
    )
    if manifest is not None:
        # Mirror what the pipeline does: the file row first (chunks carry a
        # foreign key to it), then the vectors, then the manifest rows. A
        # store-only fixture would hide the very bug this module is testing.
        for source in sources:
            manifest.record_file(source, chunker="code", chunker_version=1)
            manifest.record_space(source.root_label, source.rel_path, WIDE.slug(), 1)
        manifest.record_chunks(chunks, WIDE.slug())


def test_truncate_keeps_the_leading_entries() -> None:
    """Matryoshka: the prefix is itself a valid embedding, so which entries we
    keep is not a free choice."""
    result = truncate([3.0, 4.0, 99.0, 99.0], 2)
    assert len(result) == 2
    assert result[0] < result[1]


def test_truncate_normalises() -> None:
    """Not required for cosine, which ignores magnitude — but free, and
    necessary the moment a collection uses dot-product distance."""
    result = truncate([3.0, 4.0, 0.0, 0.0], 2)
    assert math.isclose(math.sqrt(sum(v * v for v in result)), 1.0)


def test_truncate_preserves_direction() -> None:
    original = [3.0, 4.0, 5.0, 6.0]
    shortened = truncate(original, 2)
    scale = shortened[0] / original[0]
    assert math.isclose(shortened[1], original[1] * scale)


def test_truncate_handles_a_zero_vector() -> None:
    assert truncate([0.0, 0.0, 0.0], 2) == [0.0, 0.0]


async def test_widening_is_refused(store: QdrantStore, manifest: Manifest) -> None:
    """Truncation cannot invent information; going the other way is a full
    re-embed, which is the asymmetry that decides indexing at 2048 first."""
    await _seed(store)
    with pytest.raises(ValueError, match="only truncation is free"):
        await _reprojector(store, manifest).reproject(WIDE, 16)


async def test_reproject_creates_a_separate_collection(
    store: QdrantStore, manifest: Manifest
) -> None:
    """The narrow collection must not overwrite the thing it is compared
    against."""
    await _seed(store)
    target = await _reprojector(store, manifest).reproject(WIDE, 4)
    assert target.slug() != WIDE.slug()
    names = await store.collection_names()
    assert store.collection_name(WIDE) in names
    assert store.collection_name(target) in names


async def test_every_point_is_carried_across(store: QdrantStore, manifest: Manifest) -> None:
    await _seed(store, count=5)
    target = await _reprojector(store, manifest).reproject(WIDE, 4)
    assert await store.count(target) == await store.count(WIDE) == 5


async def test_ids_are_preserved(store: QdrantStore, manifest: Manifest) -> None:
    """Same chunk, same id: the manifest keys off it in both spaces."""
    await _seed(store)
    target = await _reprojector(store, manifest).reproject(WIDE, 4)
    wide_ids = {pid async for pid, _, _ in store.scroll(WIDE)}
    narrow_ids = {pid async for pid, _, _ in store.scroll(target)}
    assert wide_ids == narrow_ids


async def test_payload_survives_except_the_space_slug(
    store: QdrantStore, manifest: Manifest
) -> None:
    """Results from either collection have to render identically."""
    await _seed(store)
    target = await _reprojector(store, manifest).reproject(WIDE, 4)
    wide = {pid: payload async for pid, payload, _ in store.scroll(WIDE)}
    narrow = {pid: payload async for pid, payload, _ in store.scroll(target)}
    for pid, payload in narrow.items():
        assert payload["space_slug"] == target.slug()
        assert payload["source_text"] == wide[pid]["source_text"]
        assert payload["rel_path"] == wide[pid]["rel_path"]
        assert payload["start_line"] == wide[pid]["start_line"]


async def test_vectors_are_the_requested_width(store: QdrantStore, manifest: Manifest) -> None:
    await _seed(store)
    target = await _reprojector(store, manifest).reproject(WIDE, 4)
    async for _, _, vectors in store.scroll(target, with_vectors=True):
        assert vectors is not None
        assert len(vectors["dense"]) == 4


async def test_sparse_vectors_are_reattached(store: QdrantStore, manifest: Manifest) -> None:
    """Dropping them would leave the narrow collection unable to do hybrid
    search, quietly making the comparison unfair."""
    await _seed(store)
    target = await _reprojector(store, manifest).reproject(WIDE, 4)
    async for _, _, vectors in store.scroll(target, with_vectors=True):
        assert vectors is not None
        assert "bm25" in vectors


async def test_the_narrow_collection_is_searchable(store: QdrantStore, manifest: Manifest) -> None:
    from workspace_indexer.storage.query_spec import QuerySpec

    await _seed(store)
    target = await _reprojector(store, manifest).reproject(WIDE, 4)
    hits = await store.search(target, QuerySpec(dense=[1.0, 0, 0, 0], fusion="dense_only"))
    assert hits


async def test_batching_covers_everything(store: QdrantStore, manifest: Manifest) -> None:
    await _seed(store, count=25)
    target = await _reprojector(store, manifest).reproject(WIDE, 4, batch_size=4)
    assert await store.count(target) == 25


async def test_reproject_is_idempotent(store: QdrantStore, manifest: Manifest) -> None:
    """Re-running after an interruption must not duplicate points."""
    await _seed(store)
    reprojector = _reprojector(store, manifest)
    target = await reprojector.reproject(WIDE, 4)
    await reprojector.reproject(WIDE, 4)
    assert await store.count(target) == 5


async def test_reprojected_space_gets_its_own_slug(store: QdrantStore, manifest: Manifest) -> None:
    """ "Asked the model for 1024" and "truncated 2048 to 1024" are different
    spaces. Sharing a slug means sharing a collection, and a partial run then
    leaves the two silently mixed."""
    await _seed(store)
    target = await _reprojector(store, manifest).reproject(WIDE, 4)
    native = WIDE.model_copy(update={"dimensions": 4})
    assert target.slug() != native.slug()
    assert target.is_derived
    assert not native.is_derived
    assert str(WIDE.dimensions) in target.slug()


async def test_manifest_records_the_reprojected_space(
    store: QdrantStore, manifest: Manifest
) -> None:
    """Writing vectors without recording them is what stranded stale content in
    a live collection: a later index could not tell which chunks existed, and
    orphan cleanup could not see them."""
    await _seed(store, count=5, manifest=manifest)
    source_slug = WIDE.slug()
    target = await _reprojector(store, manifest).reproject(WIDE, 4)
    assert manifest.chunk_count(target.slug()) == manifest.chunk_count(source_slug)


async def test_store_and_manifest_agree_after_reprojection(
    store: QdrantStore, manifest: Manifest
) -> None:
    """The invariant the divergence check in `status` looks for."""
    await _seed(store, count=5, manifest=manifest)
    target = await _reprojector(store, manifest).reproject(WIDE, 4)
    assert await store.count(target) == manifest.chunk_count(target.slug())
