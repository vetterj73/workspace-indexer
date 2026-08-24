"""A row in the files table."""

from __future__ import annotations

from pydantic import BaseModel


class FileRecord(BaseModel):
    root_label: str
    rel_path: str
    abs_path: str
    mtime_ns: int
    size: int
    sha256: str
    kind: str
    language: str | None = None
    chunker: str | None = None
    chunker_version: int = 0
    indexed_at: str = ""
