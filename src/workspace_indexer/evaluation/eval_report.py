"""Aggregate scores for a whole dataset."""

from __future__ import annotations

from pydantic import BaseModel

from workspace_indexer.evaluation.eval_result import EvalResult


class EvalReport(BaseModel):
    label: str
    limit: int
    results: list[EvalResult]

    @property
    def recall_at_k(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.recall for r in self.results) / len(self.results)

    @property
    def mrr_at_k(self) -> float:
        """Mean reciprocal rank rewards putting the right file *first*, which
        recall alone cannot see: a result at rank 10 and one at rank 1 score
        the same on recall."""
        if not self.results:
            return 0.0
        return sum(r.reciprocal_rank for r in self.results) / len(self.results)

    @property
    def misses(self) -> list[EvalResult]:
        return [r for r in self.results if r.recall < 1.0]

    @property
    def median_ms(self) -> float:
        """The typical query. Median rather than mean: one cold start or one
        retried request drags a mean over sixteen cases far enough to change
        which backend looks faster."""
        return self._percentile(0.5)

    @property
    def p95_ms(self) -> float:
        """The tail. This is the number an agent notices, because a tool it
        calls a dozen times in a task hits the tail every task."""
        return self._percentile(0.95)

    @property
    def slowest_ms(self) -> float:
        return max((r.duration_ms for r in self.results), default=0.0)

    def _percentile(self, share: float) -> float:
        timings = sorted(r.duration_ms for r in self.results)
        if not timings:
            return 0.0
        # Nearest-rank. Exact interpolation is false precision over sixteen
        # samples, and the ranking between two backends is what this decides.
        index = min(len(timings) - 1, int(round(share * (len(timings) - 1))))
        return timings[index]
