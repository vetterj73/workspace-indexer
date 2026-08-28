"""Copying a populated collection from one backend into another.

Exists because the alternative is re-embedding. Vectors are the expensive part
of an index -- this workspace's own cost 2.6M tokens -- and they are backend
neutral: the same 1024 floats mean the same thing in Qdrant and in Atlas. So
evaluating a second store, or migrating to one, should cost a scroll and a
write rather than the whole embedding bill again.

That makes it the only honest way to compare two backends. Re-embedding into
the second one would introduce a second set of vectors, and any difference in
the numbers afterwards could be the store or could be the embeddings, with no
way to tell which.

Deliberately not a method on VectorStore. Neither store should know another
exists; this reads through the protocol on one side and writes through it on
the other, which is exactly what the protocol is for.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from workspace_indexer.models import EmbeddingSpace, SparseVec
from workspace_indexer.obs.logging import get_logger
from workspace_indexer.storage.qdrant_store import DENSE_VECTOR, SPARSE_VECTOR
from workspace_indexer.storage.vector_store import VectorStore

log = get_logger("workspace_indexer.storage.mirror")


class StoreMirror:
    def __init__(self, source: VectorStore, target: VectorStore) -> None:
        self._source = source
        self._target = target

    async def mirror(
        self, space: EmbeddingSpace, *, batch_size: int = 256, overwrite: bool = False
    ) -> int:
        """Copy every point of `space` from source to target. Returns the count.

        Idempotent by construction rather than by checking: points are keyed by
        chunk id, so writing the same point twice replaces it. Re-running after
        an interrupted mirror resumes without duplicating anything, which is
        why there is no resume flag.
        """
        if overwrite:
            await self._target.drop_collection(space)
        await self._target.ensure_collection(space)

        ids: list[str] = []
        dense: list[list[float]] = []
        sparse: list[SparseVec] = []
        payloads: list[dict[str, object]] = []
        moved = 0
        skipped = 0

        async for point_id, payload, vectors in self._source.scroll(
            space, with_vectors=True, batch_size=batch_size
        ):
            if vectors is None or DENSE_VECTOR not in vectors:
                # A point with no dense vector cannot be searched for, so
                # copying it would grow the target without making anything
                # findable. Counted rather than dropped silently.
                skipped += 1
                continue
            ids.append(point_id)
            dense.append(_as_floats(vectors[DENSE_VECTOR]))
            sparse.append(_as_sparse(vectors.get(SPARSE_VECTOR)))
            # Carried across untouched, including space_slug: this is the same
            # index in a different store, not a derived one. A hit must render
            # identically whichever backend answered.
            payloads.append(dict(payload))

            if len(ids) >= batch_size:
                await self._target.upsert_points(space, ids, dense, sparse, payloads)
                moved += len(ids)
                log.info("mirror.progress", moved=moved, collection=space.slug())
                ids, dense, sparse, payloads = [], [], [], []

        if ids:
            await self._target.upsert_points(space, ids, dense, sparse, payloads)
            moved += len(ids)

        log.info(
            "mirror.done",
            collection=space.slug(),
            moved=moved,
            skipped=skipped,
            source=self._source.describe(),
            target=self._target.describe(),
        )
        return moved


def _as_floats(raw: object) -> list[float]:
    """The dense vector as plain Python floats.

    `scroll` yields whatever its backend hands back -- a list from Qdrant, a
    decoded binData vector from Atlas -- so the shape is only known at runtime
    and has to be narrowed here rather than trusted.
    """
    if not isinstance(raw, Sequence):
        raise TypeError(f"dense vector is not a sequence: {type(raw).__name__}")
    return [float(value) for value in cast("Sequence[float]", raw)]


def _as_sparse(raw: object) -> SparseVec:
    """The keyword vector, or an empty one.

    An empty vector rather than a failure because the two backends disagree
    about whether they need it at all: Qdrant requires one per point, Atlas
    ignores the argument and builds its own inverted index. Mirroring in the
    direction that does not carry one must still produce a writable point.
    """
    indices = getattr(raw, "indices", None)
    values = getattr(raw, "values", None)
    if not isinstance(indices, Sequence) or not isinstance(values, Sequence):
        return SparseVec(indices=[], values=[])
    return SparseVec(
        indices=[int(i) for i in cast("Sequence[int]", indices)],
        values=[float(v) for v in cast("Sequence[float]", values)],
    )
