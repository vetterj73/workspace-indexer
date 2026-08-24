"""Reranking turned off."""

from __future__ import annotations

from workspace_indexer.models import SearchHit


class NoopReranker:
    """The way reranking is disabled.

    An object rather than a branch: nothing downstream asks whether reranking
    is enabled, so there is no `if rerank_enabled:` threaded through the search
    path waiting to be forgotten in one place.
    """

    name = "noop"

    async def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]:
        return hits[:top_n]
