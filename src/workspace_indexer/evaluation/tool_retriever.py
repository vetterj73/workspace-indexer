"""Scoring an MCP tool, exactly as an agent would call it.

The point of measuring here rather than one layer down: a tool's document-type
filter is the whole hypothesis. Scoring the search service underneath it would
measure everything except the thing under test.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from workspace_indexer.mcp.search_response import SearchResponse

ToolCall = Callable[[str, int], Awaitable[SearchResponse]]


class ToolRetriever:
    def __init__(self, name: str, call: ToolCall) -> None:
        self.name = name
        self._call = call

    async def retrieve(self, query: str, limit: int) -> list[str]:
        response = await self._call(query, limit)
        return [result.rel_path for result in response.results]
