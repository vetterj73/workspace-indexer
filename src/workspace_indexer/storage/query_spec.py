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
    # The query as written. Carried alongside the sparse vector rather than
    # instead of it, because the two backends implement the keyword branch
    # differently: Qdrant scores the sparse vector we encoded locally, while
    # Atlas has its own inverted index and needs the words. A store uses
    # whichever of the two it can, and neither is derivable from the other.
    text: str = ""
    fusion: Literal["rrf", "dense_only", "sparse_only"] = "rrf"
    limit: int = 10
    # Candidates pulled per branch before fusion. Raising this is a much
    # cheaper way to buy recall than widening the vectors.
    prefetch_limit: int = 50
    with_source: bool = True
