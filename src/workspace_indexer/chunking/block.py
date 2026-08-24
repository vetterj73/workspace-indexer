"""An atomic span of text that must not be split further."""

from __future__ import annotations

from pydantic import BaseModel


class Block(BaseModel):
    """A paragraph, or a fenced code block treated as one unit.

    Line numbers are 1-based and inclusive, matching what an editor shows and
    what `file:line` links in search results need.
    """

    start_line: int
    end_line: int
    text: str
    # True for fenced code. Packing may place it whole or start a new chunk for
    # it, but never cut through the middle.
    atomic: bool = False
