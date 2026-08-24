"""The reranking seam.

A Protocol rather than a base class, so "reranking is off" can be a four-line
object that inherits none of the scoring machinery it would never run, and a
test double can be any object of the right shape.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from workspace_indexer.models import SearchHit


@runtime_checkable
class Reranker(Protocol):
    name: str

    async def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]: ...
