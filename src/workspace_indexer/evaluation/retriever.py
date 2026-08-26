"""What the eval harness needs from whatever it is scoring.

A seam, so a number can be attached to a *tool* rather than only to the raw
search path. "find_guidance beats plain search on the guidance cases" is a
claim about the tool an agent will actually call, and it cannot be measured
through a service the tool sits on top of.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Retriever(Protocol):
    """Returns ranked rel_paths for a query. Paths, because that is all the
    metrics look at, and it keeps a tool's response shape out of the harness."""

    name: str

    async def retrieve(self, query: str, limit: int) -> list[str]: ...
