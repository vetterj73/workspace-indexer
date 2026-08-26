"""The read path: embed, fuse, rerank, flag stale.

Every seam it uses is a protocol, so this file never learns which embedding
provider, vector store or reranker is in play.
"""

from __future__ import annotations

import time

from workspace_indexer.config import SearchSection
from workspace_indexer.embedding.embedding_service import EmbeddingService
from workspace_indexer.embedding.sparse_backend import SparseBackend
from workspace_indexer.models import EmbeddingSpace, SearchHit
from workspace_indexer.obs.logging import get_logger
from workspace_indexer.rerank.noop_reranker import NoopReranker
from workspace_indexer.rerank.reranker import Reranker
from workspace_indexer.search.matryoshka import truncate
from workspace_indexer.search.search_request import SearchRequest
from workspace_indexer.search.staleness import mark_stale
from workspace_indexer.storage.query_spec import QuerySpec
from workspace_indexer.storage.vector_store import VectorStore

log = get_logger("workspace_indexer.search")


class SearchService:
    def __init__(
        self,
        *,
        store: VectorStore,
        embeddings: EmbeddingService,
        sparse: SparseBackend,
        reranker: Reranker,
        config: SearchSection,
        space: EmbeddingSpace,
    ) -> None:
        self._store = store
        self._embeddings = embeddings
        self._sparse = sparse
        self._reranker = reranker
        self._config = config
        self._space = space

    async def search(self, request: SearchRequest) -> list[SearchHit]:
        started = time.monotonic()
        fusion = request.fusion or self._config.fusion
        limit = request.limit or self._config.default_limit
        reranker = self._reranker_for(request)

        # Retrieve deep, return shallow. The reranker only needs the right
        # chunk somewhere in the candidate set; it decides the final order.
        depth = max(limit, self._config.rerank.candidates) if _reranks(reranker) else limit

        # Only pay for the branch the fusion mode will actually use.
        dense = None
        if fusion != "sparse_only":
            dense = self._fit(await self._embeddings.embed_query(request.query))
        sparse = None
        if fusion != "dense_only":
            sparse = self._sparse.encode_query(request.query)

        hits = await self._store.search(
            self._space,
            QuerySpec(
                dense=dense,
                sparse=sparse,
                fusion=fusion,
                limit=depth,
                prefetch_limit=self._config.prefetch_limit,
            ),
            request.filters,
        )

        ranked = await reranker.rerank(request.query, hits, limit)
        if request.check_staleness:
            ranked = mark_stale(ranked)

        log.info(
            "search.query",
            query=request.query,
            fusion=fusion,
            filtered=not request.filters.is_empty(),
            candidates=len(hits),
            returned=len(ranked),
            reranker=reranker.name,
            duration_ms=round((time.monotonic() - started) * 1000, 1),
        )
        return ranked

    async def chunks_for_path(
        self, rel_path: str, limit: int = 50, *, check_staleness: bool = True
    ) -> list[SearchHit]:
        """Every indexed chunk of one file, staleness flagged.

        Not a query: an agent holding one hit and wanting the code around it
        should not have to invent a search that happens to retrieve the
        neighbours. `chunk_index`/`chunk_total` went into the payload for this.
        """
        hits = await self._store.chunks_for_path(self._space, rel_path, limit=limit)
        return mark_stale(hits) if check_staleness else hits

    def _fit(self, vector: list[float]) -> list[float]:
        """Match the query vector to the collection being searched.

        A reprojected collection is narrower than the embedder's native output,
        so the query has to be truncated the same way its documents were.
        Otherwise the vectors are not comparable at all and the store rejects
        the query outright.
        """
        target = self._space.dimensions
        if len(vector) == target:
            return vector
        if len(vector) < target:
            # Truncation cannot invent information; this means the configured
            # dimensions and the model disagree.
            raise RuntimeError(
                f"query embedding has {len(vector)} dimensions, "
                f"collection {self._space.slug()} expects {target}"
            )
        return truncate(vector, target)

    def _reranker_for(self, request: SearchRequest) -> Reranker:
        """A per-call override swaps the object rather than setting a flag, so
        the rest of this method never asks whether reranking is on."""
        if request.rerank is False:
            return NoopReranker()
        return self._reranker


def _reranks(reranker: Reranker) -> bool:
    return not isinstance(reranker, NoopReranker)
