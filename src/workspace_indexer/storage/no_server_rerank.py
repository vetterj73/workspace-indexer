"""Server-side reranking, off."""

from __future__ import annotations

from typing import Any


class NoServerRerank:
    """The default: the store retrieves and something else decides the order.

    Four lines, and none of them a branch. Every store appends `stages()`
    unconditionally; this one returns the plain scoring tail it would have
    written anyway.
    """

    name = "none"

    def depth(self, limit: int) -> int:
        """Exactly what was asked for. Widening the candidate set is only
        worth paying for when something is going to reorder it, and here
        nothing in the store is."""
        return limit

    def stages(self, query: str, limit: int, score_meta: str) -> list[dict[str, Any]]:
        # The relevance score lifted out of metadata into a real field. Named
        # per branch by the caller, because asking for the wrong one is not an
        # error -- it yields a missing field, so every hit scores 0.0 and the
        # ranking silently collapses.
        return [{"$addFields": {"score": {"$meta": score_meta}}}]
