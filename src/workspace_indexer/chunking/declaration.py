"""One named declaration and the lines it spans."""

from __future__ import annotations

from pydantic import BaseModel


class Declaration(BaseModel):
    name: str
    # "function" or "method", matching what the symbol table already uses, so
    # a caller cannot tell where the attribution came from.
    kind: str
    # 1-based and inclusive, matching every other line number we expose.
    start_line: int
    end_line: int

    def contains(self, line: int) -> bool:
        return self.start_line <= line <= self.end_line
