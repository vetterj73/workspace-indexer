"""A ScoringReranker whose scores are dictated by the test.

The real providers cannot be made to fail on demand, and a cross-encoder's
scores are not predictable enough to assert an exact ordering against.
"""

from __future__ import annotations

from workspace_indexer.config import RerankConfig
from workspace_indexer.rerank.scoring_reranker import ScoringReranker


class FakeScoringReranker(ScoringReranker):
    name = "fake"

    def __init__(
        self,
        config: RerankConfig,
        *,
        scores: list[float] | None = None,
        error: Exception | None = None,
        reverse: bool = False,
        wrong_length: bool = False,
        cost: float | None = None,
    ) -> None:
        super().__init__(config)
        self._scores = scores
        self._error = error
        self._reverse = reverse
        self._wrong_length = wrong_length
        self._cost = cost
        self.seen_queries: list[str] = []
        self.seen_documents: list[list[str]] = []

    async def _score(self, query: str, documents: list[str]) -> list[float]:
        self.seen_queries.append(query)
        self.seen_documents.append(list(documents))
        if self._error is not None:
            raise self._error
        if self._wrong_length:
            return [1.0] * (len(documents) - 1)
        if self._scores is not None:
            return list(self._scores[: len(documents)])
        if self._reverse:
            # Exactly inverts the incoming order, so a no-op bug is visible.
            return [float(i) for i in range(len(documents))]
        return [1.0] * len(documents)

    def cost_of_last_call(self) -> float | None:
        return self._cost
