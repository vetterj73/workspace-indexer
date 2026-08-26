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
    # Whether a hit is checked against the file on disk before it is returned.
    #
    # On by default, because showing text that no longer exists is worse than
    # showing a warning. But the check needs read access to the indexed source,
    # and a deployment that puts the MCP server next to Qdrant rather than next
    # to the code has none -- every hit then comes back flagged stale, which is
    # both wrong and useless, since a flag on everything carries no signal.
    #
    # Set false on such a box. The honest trade is that a result may then be
    # out of date without saying so, which is why it is not the default.
    check_staleness: bool = True
