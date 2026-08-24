"""What to ask the vector store for."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from workspace_indexer.models import SparseVec


class QuerySpec(BaseModel):
    """Both branches of a hybrid query, plus how to combine them.

    dense_only and sparse_only are debugging tools, not features: when a query
    returns junk the first question is which branch produced it.
    """

    dense: list[float] | None = None
    sparse: SparseVec | None = None
    fusion: Literal["rrf", "dense_only", "sparse_only"] = "rrf"
    limit: int = 10
    # Candidates pulled per branch before fusion. Raising this is a much
    # cheaper way to buy recall than widening the vectors.
    prefetch_limit: int = 50
    with_source: bool = True
