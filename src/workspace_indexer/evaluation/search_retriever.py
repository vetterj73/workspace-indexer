"""Scoring the raw search path -- the baseline every tool is measured against."""

from __future__ import annotations

from typing import Literal

from workspace_indexer.search.search_request import SearchRequest
from workspace_indexer.search.search_service import SearchService

Fusion = Literal["rrf", "dense_only", "sparse_only"]


class SearchRetriever:
    _fusion: Fusion | None

    def __init__(
        self,
        search: SearchService,
        *,
        fusion: Fusion | None = None,
        rerank: bool | None = None,
    ) -> None:
        self._search = search
        self._fusion = fusion
        self._rerank = rerank
        self.name = "search"

    async def retrieve(self, query: str, limit: int) -> list[str]:
        hits = await self._search.search(
            SearchRequest(
                query=query,
                limit=limit,
                fusion=self._fusion,
                rerank=self._rerank,
                # Reading every hit's file to compare text is pure overhead for
                # a measurement that only looks at paths.
                check_staleness=False,
            )
        )
        return [hit.rel_path for hit in hits]
