"""The `search:` block."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from workspace_indexer.config.rerank_config import RerankConfig
from workspace_indexer.config.strict import Strict


class SearchSection(Strict):
    # dense_only / sparse_only are debugging tools: when a query returns junk,
    # the first question is which branch produced the junk.
    fusion: Literal["rrf", "dense_only", "sparse_only"] = "rrf"
    prefetch_limit: int = 50
    default_limit: int = 10
    rerank: RerankConfig = Field(default_factory=RerankConfig)
