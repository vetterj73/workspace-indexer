"""Voyage rerank-2.5 / rerank-2.5-lite."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from workspace_indexer.config import RerankConfig
from workspace_indexer.rerank.scoring_reranker import ScoringReranker

if TYPE_CHECKING:
    from voyageai.client_async import AsyncClient

# Per the API: at most 1000 documents, query <= 8K tokens, query plus any single
# document <= 32K. None of these bind at ~512-token chunks.
MAX_DOCUMENTS = 1000


class VoyageReranker(ScoringReranker):
    name = "voyage"

    def __init__(self, config: RerankConfig, model: str, api_key: str | None = None) -> None:
        super().__init__(config)
        # From the module rather than the package root: voyageai declares
        # py.typed but no __all__, so the root re-export reads as private.
        from voyageai.client_async import AsyncClient

        self._model = model
        self._client: AsyncClient = AsyncClient(api_key=api_key)
        self._last_tokens = 0

    async def _score(self, query: str, documents: list[str]) -> list[float]:
        result = await self._client.rerank(
            query=query,
            documents=documents[:MAX_DOCUMENTS],
            model=self._model,
            # Truncate rather than reject an over-long pair: a degraded score
            # beats failing the search.
            truncation=True,
        )
        # Results come back sorted by relevance with an index into the input,
        # so they have to be scattered back into input order before the
        # template sorts them itself.
        scores = [0.0] * len(documents)
        # RerankingResult carries no annotations, so the fields are narrowed
        # here rather than trusted.
        for item in cast("list[Any]", result.results):
            scores[int(item.index)] = float(item.relevance_score)
        self._last_tokens = int(getattr(result, "total_tokens", 0) or 0)
        return scores

    def cost_of_last_call(self) -> float | None:
        # $0.02 per million tokens for both rerank-2.5 models. Latency, not
        # price, is the reason to prefer -lite.
        return self._last_tokens * 0.02 / 1_000_000
