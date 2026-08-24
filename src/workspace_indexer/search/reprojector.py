"""Derive a lower-dimension collection from vectors we already paid for.

voyage-code-4 uses Matryoshka learning, so the first 1024 entries of a 2048-d
vector are themselves a valid 1024-d embedding. That makes the
2048-versus-1024 question answerable by measurement instead of guesswork: index
once at the wider setting, truncate locally, and run the eval harness against
both. No re-embedding, no additional API spend.

The asymmetry is what decides the default: going 2048 -> 1024 is free, going
1024 -> 2048 is a full re-embed of the whole workspace.
"""

from __future__ import annotations

import math
from typing import Any, cast

from workspace_indexer.models import EmbeddingSpace, SparseVec
from workspace_indexer.obs.logging import get_logger
from workspace_indexer.storage.qdrant_store import DENSE_VECTOR, SPARSE_VECTOR, QdrantStore

log = get_logger("workspace_indexer.search.reproject")


def truncate(vector: list[float], dimensions: int) -> list[float]:
    """Take the leading entries and re-normalise.

    Cosine distance ignores magnitude, so normalising is not strictly required
    today — but it costs nothing and keeps the result correct if the collection
    is ever switched to dot-product distance, where it very much is.
    """
    head = vector[:dimensions]
    norm = math.sqrt(sum(value * value for value in head))
    if norm == 0.0:
        return head
    return [value / norm for value in head]


class Reprojector:
    def __init__(self, store: QdrantStore) -> None:
        self._store = store

    async def reproject(
        self, source: EmbeddingSpace, dimensions: int, *, batch_size: int = 256
    ) -> EmbeddingSpace:
        if dimensions >= source.dimensions:
            # Truncation cannot invent information. Widening needs a re-embed.
            raise ValueError(
                f"cannot reproject {source.dimensions} dimensions up to {dimensions}; "
                "only truncation is free"
            )

        target = source.model_copy(update={"dimensions": dimensions})
        await self._store.ensure_collection(target)

        ids: list[str] = []
        dense: list[list[float]] = []
        sparse: list[SparseVec] = []
        payloads: list[dict[str, object]] = []
        moved = 0

        async for point_id, payload, vectors in self._store.scroll(
            source, with_vectors=True, batch_size=batch_size
        ):
            if vectors is None or DENSE_VECTOR not in vectors:
                continue
            ids.append(point_id)
            dense.append(truncate(_as_floats(vectors[DENSE_VECTOR]), dimensions))
            sparse.append(_as_sparse(vectors.get(SPARSE_VECTOR)))
            # The payload is carried across untouched except for the space
            # slug, so results from either collection render identically.
            payloads.append({**payload, "space_slug": target.slug()})

            if len(ids) >= batch_size:
                await self._store.upsert_points(target, ids, dense, sparse, payloads)
                moved += len(ids)
                ids, dense, sparse, payloads = [], [], [], []

        if ids:
            await self._store.upsert_points(target, ids, dense, sparse, payloads)
            moved += len(ids)

        log.info(
            "reproject.done",
            source=source.slug(),
            target=target.slug(),
            points=moved,
            dimensions=dimensions,
        )
        return target


def _as_floats(value: object) -> list[float]:
    if not isinstance(value, list):
        raise TypeError(f"expected a dense vector, got {type(value).__name__}")
    # isinstance narrows `object` only to list[Unknown]; the elements are
    # numbers by the collection's own schema.
    return [float(item) for item in cast("list[Any]", value)]


def _as_sparse(value: object) -> SparseVec:
    indices = getattr(value, "indices", None)
    values = getattr(value, "values", None)
    if indices is None or values is None:
        return SparseVec(indices=[], values=[])
    return SparseVec(indices=[int(i) for i in indices], values=[float(v) for v in values])
