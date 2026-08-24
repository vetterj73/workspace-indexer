"""One query, plus the per-call overrides a caller may want."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from workspace_indexer.models import SearchFilters


class SearchRequest(BaseModel):
    query: str
    filters: SearchFilters = SearchFilters()
    limit: int | None = None
    # Per-call overrides of the configured defaults. dense_only and sparse_only
    # are debugging tools: when a query returns junk, the first question is
    # which branch produced it.
    fusion: Literal["rrf", "dense_only", "sparse_only"] | None = None
    rerank: bool | None = None
    # Reading every hit's file to compare hashes costs I/O the eval harness
    # does not need.
    check_staleness: bool = True
