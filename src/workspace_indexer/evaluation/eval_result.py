"""How one query scored."""

from __future__ import annotations

from pydantic import BaseModel


class EvalResult(BaseModel):
    query: str
    expected: list[str]
    found: list[str]
    # 1-based position of the first expected file, or None if it never
    # appeared. This is what MRR is computed from.
    first_hit_rank: int | None = None

    @property
    def recall(self) -> float:
        if not self.expected:
            return 0.0
        hit = sum(1 for want in self.expected if any(want in got for got in self.found))
        return hit / len(self.expected)

    @property
    def reciprocal_rank(self) -> float:
        return 0.0 if self.first_hit_rank is None else 1.0 / self.first_hit_rank
