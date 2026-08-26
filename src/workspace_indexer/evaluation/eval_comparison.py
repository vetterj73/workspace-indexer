"""The delta between two runs, per case rather than only in aggregate.

An aggregate that moved says something changed; it does not say what. The
per-case movement is what turns a number into a decision -- and a run that
improves the average while breaking two cases is a result worth seeing.
"""

from __future__ import annotations

from pydantic import BaseModel

from workspace_indexer.evaluation.case_movement import CaseMovement
from workspace_indexer.evaluation.eval_record import EvalRecord


class EvalComparison(BaseModel):
    before: EvalRecord
    after: EvalRecord
    movements: list[CaseMovement]

    @property
    def recall_delta(self) -> float:
        return self.after.recall_at_k - self.before.recall_at_k

    @property
    def mrr_delta(self) -> float:
        return self.after.mrr_at_k - self.before.mrr_at_k

    @property
    def improved(self) -> list[CaseMovement]:
        return [m for m in self.movements if m.improved]

    @property
    def regressed(self) -> list[CaseMovement]:
        return [m for m in self.movements if m.regressed]

    @property
    def comparable(self) -> bool:
        return self.after.comparable_to(self.before)


def compare(before: EvalRecord, after: EvalRecord) -> EvalComparison:
    """Matched on the query text, so reordering the dataset does not look like
    a change and a removed case simply drops out."""
    earlier = {result.query: result for result in before.results}
    movements = [
        CaseMovement(
            query=result.query,
            before_rank=earlier[result.query].first_hit_rank
            if result.query in earlier
            else None,
            after_rank=result.first_hit_rank,
        )
        for result in after.results
        if result.query in earlier
    ]
    return EvalComparison(before=before, after=after, movements=movements)
