"""A row in the files table."""

from __future__ import annotations

from pydantic import BaseModel

from workspace_indexer.models import DocumentType


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
    # Cached against sha256, so unchanged bytes are never reclassified. Here
    # rather than only in the vector payload because the dependency graph is
    # relational: "which of my callers are tests" is a join, not a search.
    doc_type: DocumentType = DocumentType.UNKNOWN
    indexed_at: str = ""
