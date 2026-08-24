"""One query and the files it should surface."""

from __future__ import annotations

from pydantic import BaseModel


class EvalCase(BaseModel):
    query: str
    # Substring match against rel_path, so a case survives a file moving
    # within a directory without needing the dataset rewritten.
    expect: list[str]
    note: str | None = None
