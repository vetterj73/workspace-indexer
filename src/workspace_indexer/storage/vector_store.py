"""The storage seam."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

from workspace_indexer.models import Chunk, EmbeddingSpace, SearchFilters, SearchHit, SparseVec
from workspace_indexer.storage.query_spec import QuerySpec


@runtime_checkable
class VectorStore(Protocol):
    """Async because the indexing pipeline is: a synchronous client would stall
    the event loop that the concurrent embedding requests depend on."""

    async def ensure_collection(self, space: EmbeddingSpace) -> None: ...

    async def upsert(
        self,
        space: EmbeddingSpace,
        chunks: Sequence[Chunk],
        dense: Sequence[Sequence[float]],
        sparse: Sequence[SparseVec],
    ) -> None: ...

    async def delete_by_ids(self, space: EmbeddingSpace, chunk_ids: Sequence[str]) -> None: ...

    async def delete_by_path(
        self, space: EmbeddingSpace, root_label: str, rel_path: str
    ) -> None: ...

    async def search(
        self, space: EmbeddingSpace, query: QuerySpec, filters: SearchFilters | None = None
    ) -> list[SearchHit]: ...

    async def describe_vectors(self, space: EmbeddingSpace) -> dict[str, list[str]]: ...

    async def count(self, space: EmbeddingSpace) -> int: ...

    async def collection_names(self) -> list[str]: ...

    def scroll(
        self, space: EmbeddingSpace, *, with_vectors: bool = False, batch_size: int = 256
    ) -> AsyncIterator[tuple[str, dict[str, object], dict[str, object] | None]]: ...

    async def drop_collection(self, space: EmbeddingSpace) -> None: ...

    async def close(self) -> None: ...
