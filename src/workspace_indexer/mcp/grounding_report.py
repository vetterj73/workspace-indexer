"""What an agent gets back when it asks whether a codebase records its reasons."""

from __future__ import annotations

from pydantic import BaseModel

from workspace_indexer.grounding import UnitCoverage


class GroundingReport(BaseModel):
    """Per-repository coverage, plus how to read it.

    The note is not decoration. This tool exists so an agent can tell an empty
    `find_guidance` result apart from a codebase that never wrote the answer
    down, and an agent that reads the numbers without that framing will draw
    the first conclusion in both cases -- which is the failure the report was
    built to prevent.
    """

    repositories: list[UnitCoverage] = []
    # Echoed so a scoped answer is self-describing; None when every repository
    # was reported.
    scoped_to: str | None = None
    note: str = ""
