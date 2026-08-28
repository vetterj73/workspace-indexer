"""Copying a collection between backends.

Qdrant to Qdrant here, which is not a compromise: the mirror only ever speaks
the protocol, so the pair of stores it is handed is exactly the variable it
does not care about. Two embedded Qdrant instances exercise every branch --
batching, the resume path, an absent vector -- with no credentials and no
index build, and `test_vector_store_contract.py` is what establishes that
Atlas behaves the same through that protocol.

What matters here is that vectors survive. A mirror that silently wrote
payloads without them would produce a target that counts correctly, renders
hits correctly, and cannot retrieve anything.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from workspace_indexer.models import EmbeddingSpace, SparseVec, compute_chunk_id
from workspace_indexer.storage.qdrant_store import QdrantStore
from workspace_indexer.storage.query_spec import QuerySpec
from workspace_indexer.storage.store_mirror import StoreMirror

SPACE = EmbeddingSpace(model="mirror:model", dimensions=4)


def _id(name: str) -> str:
    return compute_chunk_id("main", f"src/{name}.py", None, 0, name * 8)


def _payload(name: str) -> dict[str, object]:
    return {
        "workspace": "w",
        "root_label": "main",
        "unit": "src",
        "rel_path": f"src/{name}.py",
        "ancestors": ["src"],
        "kind": "code",
        "language": "python",
        "doc_type": "implementation",
        "source_text": f"body of {name}",
        "context_header": "",
        "start_line": 1,
        "end_line": 2,
        "space_slug": SPACE.slug(),
    }


async def _store(path: Path) -> QdrantStore:
    return QdrantStore(AsyncQdrantClient(path=str(path)), workspace="w", payload_indexes=False)


@pytest.fixture
async def pair(tmp_path: Path) -> AsyncIterator[tuple[QdrantStore, QdrantStore]]:
    source = await _store(tmp_path / "source")
    target = await _store(tmp_path / "target")
    try:
        yield source, target
    finally:
        await source.close()
        await target.close()


def _axis(index: int) -> list[float]:
    """A distinct direction per point.

    Scaling one axis would not do: the collection is cosine, which ignores
    magnitude, so [1,0,0,0] and [2,0,0,0] are the same vector as far as
    retrieval is concerned and "which is nearest" would have no answer.
    """
    vector = [0.0] * SPACE.dimensions
    vector[index % SPACE.dimensions] = 1.0
    return vector


async def _seed(store: QdrantStore, names: list[str]) -> None:
    await store.upsert_points(
        SPACE,
        [_id(n) for n in names],
        [_axis(i) for i, _ in enumerate(names)],
        [SparseVec(indices=[i + 1], values=[1.0]) for i, _ in enumerate(names)],
        [_payload(n) for n in names],
    )


async def test_every_point_arrives(pair: tuple[QdrantStore, QdrantStore]) -> None:
    source, target = pair
    await _seed(source, ["a", "b", "c"])

    moved = await StoreMirror(source, target).mirror(SPACE)

    assert moved == 3
    assert await target.count(SPACE) == 3


async def test_the_vectors_arrive_not_only_the_payloads(
    pair: tuple[QdrantStore, QdrantStore],
) -> None:
    """The failure worth guarding: a target that counts right, renders hits
    right, and retrieves nothing, because the payloads came across without the
    vectors."""
    source, target = pair
    await _seed(source, ["a", "b"])

    await StoreMirror(source, target).mirror(SPACE)

    hits = await target.search(
        SPACE, QuerySpec(dense=_axis(1), text="", fusion="dense_only", limit=2)
    )
    assert [h.rel_path for h in hits][0] == "src/b.py"
    assert hits[0].score > 0.0


async def test_the_payload_is_carried_across_untouched(
    pair: tuple[QdrantStore, QdrantStore],
) -> None:
    """Same index in a different store, not a derived one: a hit must render
    identically whichever backend answered, `space_slug` included."""
    source, target = pair
    await _seed(source, ["a"])

    await StoreMirror(source, target).mirror(SPACE)

    original = (await source.chunks_for_path(SPACE, "src/a.py"))[0]
    copied = (await target.chunks_for_path(SPACE, "src/a.py"))[0]
    assert copied.model_dump() == original.model_dump()


async def test_more_points_than_one_batch(pair: tuple[QdrantStore, QdrantStore]) -> None:
    """The flush inside the loop and the flush after it are different code
    paths, and a corpus smaller than one batch never reaches the first."""
    source, target = pair
    names = [f"f{i}" for i in range(7)]
    await _seed(source, names)

    moved = await StoreMirror(source, target).mirror(SPACE, batch_size=3)

    assert moved == 7
    assert await target.count(SPACE) == 7


async def test_re_running_replaces_rather_than_duplicating(
    pair: tuple[QdrantStore, QdrantStore],
) -> None:
    """Why there is no resume flag: points are keyed by chunk id, so an
    interrupted mirror is resumed by running it again."""
    source, target = pair
    await _seed(source, ["a", "b"])
    mirror = StoreMirror(source, target)

    await mirror.mirror(SPACE)
    await mirror.mirror(SPACE)

    assert await target.count(SPACE) == 2


async def test_overwrite_drops_what_the_target_already_held(
    pair: tuple[QdrantStore, QdrantStore],
) -> None:
    """A point deleted from the source is not deleted from the target by a
    plain mirror -- nothing scrolls past it to say so. `--overwrite` is the
    only way to make the target match rather than merely include."""
    source, target = pair
    await _seed(source, ["a"])
    await _seed(target, ["stale"])

    await StoreMirror(source, target).mirror(SPACE, overwrite=True)

    assert [h.rel_path for h in await target.chunks_for_path(SPACE, "src/stale.py")] == []
    assert await target.count(SPACE) == 1


async def test_mirroring_an_empty_collection_is_not_an_error(
    pair: tuple[QdrantStore, QdrantStore],
) -> None:
    source, target = pair
    await source.ensure_collection(SPACE)

    assert await StoreMirror(source, target).mirror(SPACE) == 0
    assert await target.count(SPACE) == 0
