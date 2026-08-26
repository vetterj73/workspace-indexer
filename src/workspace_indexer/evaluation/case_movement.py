"""How one eval case moved between two runs."""

from __future__ import annotations

from pydantic import BaseModel


class CaseMovement(BaseModel):
    query: str
    before_rank: int | None
    after_rank: int | None

    @property
    def improved(self) -> bool:
        if self.after_rank is None:
            return False
        return self.before_rank is None or self.after_rank < self.before_rank

    @property
    def regressed(self) -> bool:
        if self.before_rank is None:
            return False
        return self.after_rank is None or self.after_rank > self.before_rank

    def __str__(self) -> str:
        def render(rank: int | None) -> str:
            return "none" if rank is None else str(rank)

        return f"{render(self.before_rank)} -> {render(self.after_rank)}  {self.query}"
