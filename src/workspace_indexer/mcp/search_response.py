"""Results, plus what it took to fit them."""

from __future__ import annotations

from pydantic import BaseModel

from workspace_indexer.mcp.search_result import SearchResult


class SearchResponse(BaseModel):
    results: list[SearchResult] = []
    # What the search actually did, in the agent's own vocabulary. An empty
    # result set is ambiguous on its own -- "nothing matched" and "your filter
    # excluded everything" call for opposite next moves.
    query: str = ""
    applied_filters: dict[str, str] = {}
    total_matches: int = 0
    returned: int = 0
    # Hits dropped to stay inside the token budget. Reported rather than
    # silently swallowed: a truncated list that looks complete is how an agent
    # concludes it has seen everything when it has seen the first three.
    dropped_for_budget: int = 0
    # Plain-language guidance when the result set is empty or clipped. This is
    # what stops an empty response from being read as "the workspace has none".
    note: str | None = None
