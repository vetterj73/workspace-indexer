"""One MCP tool call and what it returned."""

from __future__ import annotations

from pydantic import BaseModel


class ToolCall(BaseModel):
    tool: str
    query: str
    # Echoed back as the caller gave them, so a recorded call can be replayed
    # exactly. Rendered as one string because the set differs per tool and a
    # column per parameter would be a migration every time a tool grows one.
    parameters: dict[str, str] = {}
    # Paths only. Recording source_text would duplicate the index into the
    # manifest, and paths plus ranks are what an eval actually scores.
    returned_paths: list[str] = []
    total_matches: int = 0
    dropped_for_budget: int = 0
    # Present when the response carried an explanation -- an empty result set,
    # or a list clipped to fit. The two call for opposite next moves, so which
    # one happened has to survive.
    note: str | None = None
    duration_ms: float = 0.0

    @property
    def returned(self) -> int:
        return len(self.returned_paths)

    @property
    def disappointed(self) -> bool:
        """Returned nothing, or had to drop results to fit.

        The calls worth turning into eval cases: either the index had no answer
        or it had more than it could hand back.
        """
        return not self.returned_paths or self.dropped_for_budget > 0
