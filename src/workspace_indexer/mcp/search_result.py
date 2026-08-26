"""One hit, shaped for an agent rather than a terminal."""

from __future__ import annotations

from pydantic import BaseModel


class SearchResult(BaseModel):
    # `path:start-end`, so the next action is `Read(file, offset, limit)` with
    # nothing to work out. The single most valuable field here.
    location: str
    rel_path: str
    start_line: int
    end_line: int
    doc_type: str
    # Class/method trail or heading path. Tells the agent what it is looking at
    # before it reads a line of the body.
    symbol_path: str | None = None
    language: str | None = None
    repo: str | None = None
    text: str = ""
    # Set when `text` was cut to fit the budget, so the agent knows to read the
    # file rather than assuming it has the whole chunk.
    text_truncated: bool = False
    # The file changed after it was indexed: this text matched the query, but
    # it is not what is on disk now. Never silently hidden -- an agent editing
    # from stale text writes a patch that will not apply.
    stale: bool = False
    score: float = 0.0
