"""Everything every reranker does, written once.

Providers differ in exactly one respect: how they turn a query and a list of
documents into a score per document. Voyage returns reordered results with
indices, a local cross-encoder returns scores in input order — both reduce to
the same abstract method, so API and local implementations share this template
rather than living in separate hierarchies.

Kept below the Reranker protocol on purpose. If the interface carried these
implementations, NoopReranker would inherit candidate capping, instruction
handling and degrade logic it never runs.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from workspace_indexer.config import RerankConfig
from workspace_indexer.models import SearchHit
from workspace_indexer.obs.logging import get_logger
from workspace_indexer.rerank.rerank_stats import RerankStats

log = get_logger("workspace_indexer.rerank")


class ScoringReranker(ABC):
    name = "scoring"

    def __init__(self, config: RerankConfig) -> None:
        self._config = config
        self.stats = RerankStats()

    async def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]:
        if not hits:
            return []

        candidates = hits[: self._config.candidates]
        documents = [self._text_of(hit) for hit in candidates]
        prompt = self._prompt(query)

        started = time.monotonic()
        try:
            scores = await self._score(prompt, documents)
            if len(scores) != len(documents):
                # Same class of failure as a short embedding batch: silently
                # misaligned scores would reorder results by nothing at all.
                raise RuntimeError(
                    f"{self.name} returned {len(scores)} scores for {len(documents)} documents"
                )
        except Exception as exc:
            return self._degrade(exc, hits, top_n)

        ranked = self._attach(candidates, scores)
        self._record(hits, ranked, started, len(documents))
        return ranked[:top_n]

    @abstractmethod
    async def _score(self, query: str, documents: list[str]) -> list[float]:
        """One relevance score per document, in the order given."""

    def cost_of_last_call(self) -> float | None:
        """Overridden by providers that report it. None means unknown."""
        return None

    def _text_of(self, hit: SearchHit) -> str:
        # embed_text by default: the reranker benefits from the same context
        # header the embedder got, since a bare method body is ambiguous.
        if self._config.rerank_text == "source_text":
            return hit.source_text
        return hit.embed_text or hit.source_text

    def _prompt(self, query: str) -> str:
        """rerank-2.5* follow instructions but expose no instruction parameter,
        so the instruction is prepended client-side."""
        instruction = (self._config.instruction or "").strip()
        return f"{instruction}\n{query}" if instruction else query

    @staticmethod
    def _attach(candidates: list[SearchHit], scores: list[float]) -> list[SearchHit]:
        scored = [
            hit.model_copy(update={"rerank_score": float(score)})
            for hit, score in zip(candidates, scores, strict=True)
        ]
        # Stable sort, so equal scores keep the fusion order rather than
        # shuffling between identical runs.
        return sorted(scored, key=lambda h: h.rerank_score or 0.0, reverse=True)

    def _degrade(self, error: Exception, hits: list[SearchHit], top_n: int) -> list[SearchHit]:
        self.stats.degraded += 1
        if self._config.on_error == "fail":
            # Only the eval harness wants this: a silent degradation there
            # would quietly corrupt a measurement.
            log.error("error.rerank_failed", reranker=self.name, error=str(error))
            raise error
        log.warning(
            "rerank.degraded",
            reranker=self.name,
            error=f"{type(error).__name__}: {error}",
            detail="returning the fusion ordering",
        )
        return hits[:top_n]

    def _record(
        self,
        original: list[SearchHit],
        ranked: list[SearchHit],
        started: float,
        documents: int,
    ) -> None:
        churn = next(
            (i for i, hit in enumerate(original) if hit.chunk_id == ranked[0].chunk_id), -1
        )
        if churn == 0:
            self.stats.unchanged_top += 1
        cost = self.cost_of_last_call()
        self.stats.calls += 1
        self.stats.documents += documents
        if cost is not None:
            self.stats.est_cost_usd += cost
        log.info(
            "rerank.call",
            reranker=self.name,
            model=self._config.model,
            documents=documents,
            duration_ms=round((time.monotonic() - started) * 1000, 1),
            # How far the new top result moved. Consistently 0 means we are
            # paying a round trip per search and changing nothing.
            top_churn=churn,
            cost_usd=cost,
        )
