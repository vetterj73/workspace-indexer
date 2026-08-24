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
